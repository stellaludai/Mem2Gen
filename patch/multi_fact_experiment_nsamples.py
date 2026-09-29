"""Cross-layer self-patching grid for the chaining questions of one trained multi-fact run.

For every question, blocks.{tgt}.hook_resid_post at the head-entity tokens is overwritten
with blocks.{src}.hook_resid_post from the same clean pass, for every (src, tgt) layer pair.
Output: an (N, 2, L, L) array; channel 0 is the MRR of the first answer token, channel 1 is
greedy exact match. The diagonal (src == tgt) is a no-op patch. Under torchrun, rank r handles
questions r, r + world_size, ...; rows are saved rank by rank and <out>_counts.npy holds the
number of rows per rank.

Usage:
    torchrun --nproc_per_node=4 patch/multi_fact_experiment_nsamples.py --ckpt_dir <run>
    (single GPU: python patch/multi_fact_experiment_nsamples.py --ckpt_dir <run>)
    python patch/compute_oracle.py <run>/patching_results_entity_offset0.npy
"""
import os
import re
import glob
import json
import torch
import gc
import argparse
import yaml
from transformers import AutoTokenizer
from layer_patching import load_checkpoint, cross_layer_self_patch, find_entity_positions, cross_layer_self_patch_generate,CHAINING_TEMPLATES
from utils import get_prediction_metrics, pin_chat_date
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import datetime

def find_memorization_epochs(log_path, tgt_task_name):
    """Find memorization and generalization epochs from training logs."""
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    memorize_epoch, generalize_epoch = None, None
    for epoch_log in logs:
        if memorize_epoch is None and epoch_log.get('eval/memorization') == 1.0:
            memorize_epoch = epoch_log['train/epoch']
        if generalize_epoch is None and f'eval/{tgt_task_name}' in epoch_log and epoch_log.get(f'eval/{tgt_task_name}') == 1.0:
            generalize_epoch = epoch_log['train/epoch']
    
    return memorize_epoch, generalize_epoch


def read_base_model(ckpt_dir):
    """Base model name (model.name) from the training config saved in the run directory."""
    config_path = os.path.join(ckpt_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found; pass --base_model")
    with open(config_path, 'r') as f:
        name = (yaml.safe_load(f).get('model') or {}).get('name')
    if not name:
        raise ValueError(f"No model.name in {config_path}; pass --base_model")
    return name


def find_checkpoint(ckpt_dir):
    """checkpoint-last-epoch50 if present, else the checkpoint-last-epoch* with the highest epoch."""
    checkpoint_path = os.path.join(ckpt_dir, "checkpoint-last-epoch50")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    epochs = {}
    for path in glob.glob(os.path.join(ckpt_dir, "checkpoint-last-epoch*")):
        m = re.fullmatch(r"checkpoint-last-epoch(\d+)", os.path.basename(path))
        if m:
            epochs[int(m.group(1))] = path
    if not epochs:
        raise FileNotFoundError(f"No checkpoint-last-epoch* in {ckpt_dir}; pass --checkpoint")
    return epochs[max(epochs)]


def run_patching_experiment(
    ckpt_dir: str,
    base_model_name: str,
    task_name: str,
    device: str = 'cuda',
    metric_type: str = 'mrr',
    patching_position_type: str = 'entity',
    world_size: int = 1,
    ntest: int = 1,
    src_task_name: str = None,
    offset: int = 0,
    tl_model_name: str = None,
    max_instances: int = None,
    checkpoint_path: str = None,
):
    """
    Run cross-layer self-patching experiment for a single checkpoint.
    
    Args:
        ckpt_dir: Run directory (experiment_data.json, original_fact.json)
        base_model_name: Name of base model (e.g., "Qwen/Qwen2.5-3B-Instruct")
        task_name: Target task name (e.g., 'chaining')
        device: CUDA device to use
        metric_type: Metric type for evaluation ('mrr' or 'logit')
        patching_position_type: Tokens to patch ('entity' = head entity of the first fact)
        world_size: Number of ranks; rank r takes questions r, r + world_size, ...
        ntest: Prompts per question in eval_data
        src_task_name: Source prompt task, if different from task_name
        offset: Which of the ntest prompts to use
        tl_model_name: Override model name for the transformer_lens config lookup
        max_instances: Use only the first max_instances questions (None = all)
        checkpoint_path: LoRA checkpoint to merge into the base model (None = find_checkpoint(ckpt_dir))
    
    Returns:
        Dictionary containing experiment results and metadata
    """
    print(f"\n{'='*60}")
    print(f"Running experiment on: {ckpt_dir}")
    print(f"Task: {task_name}, Device: {device}")
    print(f"{'='*60}\n")
    
    # Load experiment data
    data_file = f"{ckpt_dir}/experiment_data.json"
    with open(data_file, "r") as f:
        experiment_data = json.load(f)
    
    original_fact_path = os.path.join(ckpt_dir, 'original_fact.json')
    with open(original_fact_path, 'r') as f:
        original_fact_all = json.load(f)

    # Find memorization epoch (only printed)
    log_path = os.path.join(ckpt_dir, 'training_logs.json')
    memorize_epoch, generalize_epoch = None, None
    if os.path.exists(log_path):
        memorize_epoch, generalize_epoch = find_memorization_epochs(log_path, task_name)
    
    print(f"Memorization achieved at epoch: {memorize_epoch}")
    print(f"Generalization achieved at epoch: {generalize_epoch}")

    # Load model and tokenizer
    checkpoint_path = checkpoint_path or find_checkpoint(ckpt_dir)
    print(f"Base model: {base_model_name}, checkpoint: {checkpoint_path}")
    tokenizer = pin_chat_date(AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True))
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None
    
    ckpt_tgt = load_checkpoint(
        checkpoint_path=checkpoint_path,
        base_model_name=base_model_name,
        device=device,
        tl_model_name=tl_model_name
    )

    # Prepare input
    src, tgt = {}, {}
    src['model'] = ckpt_tgt
    tgt['model'] = ckpt_tgt

    n_layers = ckpt_tgt.cfg.n_layers

    patching_results = []
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    src_task_name = task_name if src_task_name is None else src_task_name # e.g. patch from the memorization prompt into the chaining prompt

    # Map task arg to eval_data key (fact_checking_tasks -> fact_checking if needed)
    eval_task_key = task_name
    if task_name not in experiment_data['eval_data'] and task_name == 'fact_checking_tasks':
        eval_task_key = 'fact_checking'
    eval_src_key = src_task_name
    if src_task_name not in experiment_data['eval_data'] and src_task_name == 'fact_checking_tasks':
        eval_src_key = 'fact_checking'

    n_questions = len(experiment_data['eval_data'][eval_task_key]) // ntest
    if max_instances is not None:
        n_questions = min(n_questions, max_instances)
    for idx in range(local_rank, n_questions, world_size): # shard questions across ranks
        print(f"\n--- Processing example {idx + 1}/{n_questions} ---")
        original_fact = original_fact_all[idx]['facts']
        tgt['msg'] = experiment_data['eval_data'][eval_task_key][idx * ntest + offset] # a question may have ntest prompts (e.g. fact checking); offset picks one
        if src_task_name == task_name:
            src['msg'] = experiment_data['eval_data'][eval_src_key][idx * ntest + offset]
        else: # memorization task, each chaining has 2 for now
            assert len(experiment_data['eval_data'][eval_src_key]) == 2 * len(experiment_data['eval_data'][eval_task_key]), \
                "Current only support 2 hop chaining"
            if original_fact[0]['head'] in experiment_data['eval_data'][eval_src_key][idx * 2]['prompt'][1]['content']:
                src['msg'] = experiment_data['eval_data'][eval_src_key][idx * 2]
            elif original_fact[0]['head'] in experiment_data['eval_data'][eval_src_key][idx * 2 + 1]['prompt'][1]['content']:
                src['msg'] = experiment_data['eval_data'][eval_src_key][idx * 2 + 1]
            else:
                raise ValueError("Cannot find matching memorization prompt for source task.")

        # Target
        tgt['p_formatted'] = tokenizer.apply_chat_template(tgt['msg']['prompt'], tokenize=False, add_generation_prompt=True)
        tgt['answer'] = tgt['msg']['completion'][0]['content']
        tgt['p_formatted_ans'] = tgt['p_formatted'] + tgt['answer']
        tgt['tokens'] = tgt['model'].to_tokens(tgt['p_formatted_ans'], prepend_bos=False)
        tgt['answer_tokens'] = tgt['model'].to_tokens(tgt['answer'], prepend_bos=False)[0].cpu().tolist()
        tgt['tokens_prompt_len'] = len(tgt['model'].to_tokens(tgt['p_formatted'], prepend_bos=False)[0])
        
        # Source
        src['p_formatted'] = tokenizer.apply_chat_template(src['msg']['prompt'], tokenize=False, add_generation_prompt=True)
        src['answer'] = src['msg']['completion'][0]['content']
        src['p_formatted_ans'] = src['p_formatted'] + src['answer']
        src['tokens'] = src['model'].to_tokens(src['p_formatted_ans'], prepend_bos=False)
        src['answer_tokens'] = src['model'].to_tokens(src['answer'], prepend_bos=False)[0].cpu().tolist()
        src['tokens_prompt_len'] = len(src['model'].to_tokens(src['p_formatted'], prepend_bos=False)[0])

        # Get caches
        with torch.no_grad():
            logits_msrc_psrc, cache_msrc_psrc = src['model'].run_with_cache(src['tokens'])
        with torch.no_grad():
            logits_msrc_ptgt, cache_msrc_ptgt = src['model'].run_with_cache(tgt['tokens'])
        with torch.no_grad():
            logits_mtgt_ptgt, cache_mtgt_ptgt = tgt['model'].run_with_cache(tgt['tokens'])

        baseline_mtgt_ptgt = get_prediction_metrics(
            logits_mtgt_ptgt[:, :tgt['tokens_prompt_len'] + 1, :],
            tgt['answer_tokens'][0],
            metric_type
        )
        if baseline_mtgt_ptgt >= 1.0:
            #print("Warning: Baseline performance already achieved 1.0, skipping this example.")
            #patching_results.append(np.stack([np.ones((n_layers, n_layers)), np.ones((n_layers, n_layers))], axis=0))  
            need_first_self_patching = False
        else:
            need_first_self_patching = True
            
        print(f"Baseline tgt model on tgt prompt MRR: {baseline_mtgt_ptgt:.4f}")

        # Find entity positions
        if patching_position_type == 'bos':
            entity_positions_tgt = [0]
            entity_positions_src = [0]
        elif patching_position_type == 'eos':
            entity_positions_tgt = [tgt['tokens_prompt_len'] - 1]
            entity_positions_src = [src['tokens_prompt_len'] - 1]
        elif 'random' in patching_position_type:
            entity_positions_tgt = np.random.choice(
                range(tgt['tokens_prompt_len']),
                size=1,
                replace=False
            ).tolist()
            entity_positions_src = entity_positions_tgt # random: same position in src and tgt
        else:
            if patching_position_type == 'entity':
                entity_str = original_fact[0]['head']
            elif patching_position_type == 'relation1' or patching_position_type == 'relation2':
                assert task_name == 'chaining', "relation1 and relation2 only valid for chaining task"
                exp_key = (original_fact[0]['head_type'], original_fact[0]['relation'],
                        original_fact[0]['tail_type'], original_fact[1]['relation'],
                        original_fact[1]['tail_type'])
                entity_str = CHAINING_TEMPLATES[exp_key][patching_position_type]
            elif patching_position_type == 'entity3':
                assert task_name == 'intersection', "entity3 only valid for intersection task"
                entity_str = [fact['head'] for fact in original_fact[1:] if fact['tail'] == original_fact[0]['tail']][0]

            print(f"Entity: {entity_str}")
            
            entity_positions_tgt = find_entity_positions(
                tgt['model'],
                prompt=tgt['p_formatted_ans'],
                entity=entity_str
            )
            entity_positions_src = find_entity_positions(
                src['model'],
                prompt=src['p_formatted_ans'],
                entity=entity_str
            )

        print(f"Entity positions in tgt prompt: {entity_positions_tgt}")
        print(f"Entity positions in src prompt: {entity_positions_src}")
        
        # Align caches
        for i in range(n_layers):
            cache_msrc_ptgt['blocks.' + str(i) + '.hook_resid_post'][:, entity_positions_tgt, :] = \
                cache_msrc_psrc['blocks.' + str(i) + '.hook_resid_post'][:, entity_positions_src, :]
        
        # Run patching experiments
        if need_first_self_patching:
            print("\n=== Running cross-layer self-patching ===")
            results_patching_self = np.zeros((n_layers, n_layers))

            for source_layer in tqdm(range(n_layers), desc="Patching"):
                for target_layer in range(n_layers):
                    patched_logits = cross_layer_self_patch(
                        source_cache=cache_msrc_ptgt,
                        target_model=tgt['model'],
                        tokens=tgt['tokens'],
                        source_layer_id=source_layer,
                        target_layer_id=target_layer,
                        positions=entity_positions_tgt,
                        patch_type='residual'
                    )
                    patched_metric = get_prediction_metrics(
                        patched_logits[:, :tgt['tokens_prompt_len'] + 1, :],
                        tgt['answer_tokens'][0],
                        metric_type
                    )
                    results_patching_self[source_layer, target_layer] = patched_metric

        else:
            results_patching_self = np.ones((n_layers, n_layers))

        # Generation results
        print("\n=== Running generation tests ===")
        results_generation = np.zeros((n_layers, n_layers))
        
        for source_layer in tqdm(range(n_layers), desc="Generation"):
            for target_layer in range(n_layers):
                if results_patching_self[source_layer, target_layer] < 1.0: # if first patching not successful, generation must also fail
                    continue
                generated_ids = cross_layer_self_patch_generate(
                    source_cache=cache_msrc_ptgt,
                    target_model=tgt['model'],
                    prompt_tokens=tgt['model'].to_tokens(tgt['p_formatted'], prepend_bos=False),
                    source_layer_id=source_layer,
                    target_layer_id=target_layer,
                    positions=entity_positions_tgt,
                    patch_type='residual',
                    max_new_tokens=64
                )
                generated_text = tgt['model'].to_string(generated_ids[0][:-1])
                generated_ans = generated_text[len(tgt['p_formatted']):]
                
                if generated_ans.strip() == tgt['answer'].strip():
                    results_generation[source_layer, target_layer] = 1.0
        patching_results.append(np.stack([results_patching_self, results_generation], axis=0))  
    
    # Clean up
    del ckpt_tgt, src, tgt
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'ckpt_dir': ckpt_dir,
        'task_name': task_name,
        'memorize_epoch': memorize_epoch,
        'generalize_epoch': generalize_epoch,
        'patching_results': patching_results
    }

def gather_and_save(
        results, 
        out_path,
        rank,
        world_size):
    """
    results: this rank's list[np.ndarray]
             list lengths may differ across ranks, but every ndarray has the same shape.
    Rows are saved rank by rank; the per-rank counts go to *_counts.npy.
    """
    if not (dist.is_available() and dist.is_initialized()):
        # Single process: save directly
        merged = np.stack(results, axis=0) if len(results) > 0 else np.empty((0,))
        np.save(out_path, merged)
        return

    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(results, object_gather_list=gathered, dst=0)

    if rank == 0:
        # Flatten into one list
        merged_list = []
        counts = []
        for part in gathered:
            counts.append(len(part))
            merged_list.extend(part)

        # All arrays share one shape: stack to (total, *shape)
        if len(merged_list) > 0:
            merged_arr = np.stack(merged_list, axis=0)
        else:
            merged_arr = np.empty((0,))

        # Also save each rank's length so rank boundaries can be recovered
        np.save(out_path, merged_arr)
        np.save(out_path.replace(".npy", "_counts.npy"), np.array(counts, dtype=np.int64))
        print(f"\n✓ Results saved to {out_path}")

    dist.barrier()  # make sure rank 0 has finished writing before exiting

def main():
    parser = argparse.ArgumentParser(description="Run cross-layer self-patching experiments")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Run directory written by train/train_multi_fact.py")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (default: model.name in <ckpt_dir>/config.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA checkpoint dir, relative to --ckpt_dir or absolute "
                             "(default: checkpoint-last-epoch50, else the latest checkpoint-last-epoch*)")
    parser.add_argument("--task", type=str, default="chaining",
                        choices=['chaining', 'counting', 'intersection', 'fact_checking_tasks'],
                        metavar="TASK", help="Target task name (default: chaining)")
    parser.add_argument("--device", type=str, default="cuda", help="CUDA device")
    parser.add_argument("--metric", type=str, default="mrr", help="Metric type")
    parser.add_argument("--patching_position_type", type=str, default="entity", 
                        help="Type of position to patch")
    parser.add_argument("--src_task", type=str, default='', 
                        help="Source task name (if different from target task)")
    parser.add_argument("--tl_model_name", type=str, default=None,
                        help="Override model name for transformer_lens config lookup (e.g. microsoft/Phi-3-mini-4k-instruct for Phi-3.5)")
    parser.add_argument("--max_instances", type=int, default=None, metavar="N",
                        help="Use only the first N questions (default: all)")
    parser.add_argument("--out", type=str, default=None, metavar="PATH",
                        help="Output .npy path (default: <ckpt_dir>/patching_results_<position>_offset<k>.npy)")
    args = parser.parse_args()

    base_model = args.base_model or read_base_model(args.ckpt_dir)
    if args.checkpoint:
        checkpoint_path = os.path.join(args.ckpt_dir, args.checkpoint)
    else:
        checkpoint_path = find_checkpoint(args.ckpt_dir)

    # fact_checking_tasks: use the second prompt of each question
    offset = 1 if args.task == 'fact_checking_tasks' else 0

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="gloo", 
                                init_method="env://",
                                timeout=datetime.timedelta(seconds=14400),) # 4 h, so slow ranks can still gather
        args.device = f"cuda:{os.environ['LOCAL_RANK']}"

    print(f"World Size: {world_size}, Using device: {args.device}")

    if args.task == 'fact_checking_tasks':
        ntest = 2
    else:
        ntest = 1

    src_task = args.src_task if args.src_task != '' else None

    output_path = args.out or os.path.join(
        args.ckpt_dir,
        f"patching_results_{args.patching_position_type}_{src_task}_offset{offset}.npy" if src_task else f"patching_results_{args.patching_position_type}_offset{offset}.npy"
    )
    if not output_path.endswith(".npy"):
        output_path += ".npy"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    result = run_patching_experiment(
        ckpt_dir=args.ckpt_dir,
        base_model_name=base_model,
        task_name=args.task,
        device=args.device,
        metric_type=args.metric,
        patching_position_type=args.patching_position_type,
        world_size=world_size,
        ntest = ntest,
        src_task_name=src_task,
        offset=offset,
        tl_model_name=args.tl_model_name,
        max_instances=args.max_instances,
        checkpoint_path=checkpoint_path
    )
    
    if result:
        print(f"\n{'='*60}")
        print("Experiment completed successfully!")
        print(f"{'='*60}")
    else:
        print("\nExperiment failed!")
        exit(1)
    
    # Gather and save results
    gather_and_save(
        results=result['patching_results'],
        out_path=output_path,
        rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=world_size
    )


if __name__ == "__main__":
    main()