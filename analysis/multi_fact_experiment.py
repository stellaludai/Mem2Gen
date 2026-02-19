# multi_fact_experiment.py - 改进版本支持命令行参数
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import gc
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
from layer_patching import load_checkpoint, cross_layer_self_patch, find_entity_positions, cross_layer_self_patch_generate,CHAINING_TEMPLATES
from utils import format_chat_for_inference, get_prediction_metrics
import numpy as np
from tqdm import tqdm


def find_memorization_epochs(log_path, tgt_task_name):
    """Find memorization and generalization epochs from training logs."""
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    memorize_epoch, generalize_epoch = None, None
    for epoch_log in logs:
        if memorize_epoch is None and epoch_log.get('eval/memorization') == 1.0:
            memorize_epoch = epoch_log['train/epoch']
        if generalize_epoch is None and epoch_log.get(f'eval/{tgt_task_name}') == 1.0:
            generalize_epoch = epoch_log['train/epoch']
    
    return memorize_epoch, generalize_epoch


def run_patching_experiment(
    ckpt_dir: str,
    base_model_name: str,
    task_name: str,
    device: str = 'cuda:0',
    metric_type: str = 'mrr',
    patching_position_type: str = 'entity',
):
    """
    Run cross-layer self-patching experiment for a single checkpoint.
    
    Args:
        ckpt_dir: Path to checkpoint directory
        base_model_name: Name of base model (e.g., "Qwen/Qwen2.5-3B-Instruct")
        task_name: Target task name (e.g., 'chaining', 'counting', 'intersection')
        device: CUDA device to use
        metric_type: Metric type for evaluation ('mrr', 'accuracy', etc.)
        epoch_offset: Offset to add to memorization epoch when loading checkpoint
    
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
        original_fact = json.load(f)[0]['facts']

    # Find memorization epoch
    log_path = os.path.join(ckpt_dir, 'training_logs.json')
    memorize_epoch, generalize_epoch = find_memorization_epochs(log_path, task_name)
    
    print(f"Memorization achieved at epoch: {memorize_epoch}")
    print(f"Generalization achieved at epoch: {generalize_epoch}")
    
    if memorize_epoch is None:
        print("Warning: Memorization not achieved!")
        return None
    if generalize_epoch is not None:
        print("Already achieved generalization; skipping experiment.")
        return None

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True)
    # checkpoint_path = os.path.join(ckpt_dir, f"checkpoint-epoch{int(memorize_epoch) + epoch_offset}")
    checkpoint_path = os.path.join(ckpt_dir, f"checkpoint-epoch30")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None
    
    ckpt_tgt = load_checkpoint(
        checkpoint_path=checkpoint_path,
        base_model_name=base_model_name,
        device=device
    )

    # Prepare input
    src, tgt = {}, {}
    src['model'] = ckpt_tgt
    tgt['model'] = ckpt_tgt

    tgt['msg'] = experiment_data['eval_data'][task_name][0]
    src['msg'] = experiment_data['eval_data'][task_name][0]
    
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

    # Test model performance
    print("=== Testing src model on target prompt ===")
    print('Prompt tgt:', tgt['msg'])
    with torch.no_grad():
        output = src['model'].generate(
            tokenizer.encode(tgt['p_formatted'], return_tensors='pt', add_special_tokens=False).to(device),
            max_new_tokens=64,
            do_sample=False
        )[0]
        print(tokenizer.decode(output)[len(tgt['p_formatted']):])

    print("=== Testing tgt model on target prompt ===")
    with torch.no_grad():
        output = tgt['model'].generate(
            tokenizer.encode(tgt['p_formatted'], return_tensors='pt', add_special_tokens=False).to(device),
            max_new_tokens=64,
            do_sample=False
        )[0]
        print(tokenizer.decode(output)[len(tgt['p_formatted']):])

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
        entity_positions_src = entity_positions_tgt # random 选择同一个位置。
    else:
        if patching_position_type == 'entity':
            entity_str = original_fact[0]['head']
        elif patching_position_type == 'relation1' or patching_position_type == 'relation2':
            exp_key = (original_fact[0]['head_type'], original_fact[0]['relation'],
                    original_fact[0]['tail_type'], original_fact[1]['relation'],
                    original_fact[1]['tail_type'])
            entity_str = CHAINING_TEMPLATES[exp_key][patching_position_type]

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

    n_layers = ckpt_tgt.cfg.n_layers
    
    # Align caches
    for i in range(n_layers):
        cache_msrc_ptgt['blocks.' + str(i) + '.hook_resid_post'][:, entity_positions_tgt, :] = \
            cache_msrc_psrc['blocks.' + str(i) + '.hook_resid_post'][:, entity_positions_src, :]
    
    # Run patching experiments
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

    # Generation results
    print("\n=== Running generation tests ===")
    results_generation = np.zeros((n_layers, n_layers))
    
    for source_layer in tqdm(range(n_layers), desc="Generation"):
        for target_layer in range(n_layers):
            if results_patching_self[source_layer, target_layer] < 1.0:
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

    # Save results
    output_dir = Path(ckpt_dir)
    np.save(output_dir / f"results_self_patch_{patching_position_type}.npy", results_patching_self)
    np.save(output_dir / f"results_self_generation_{patching_position_type}.npy", results_generation)
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"  - results_self_patch.npy")
    print(f"  - results_self_generation.npy")
    
    # Clean up
    del ckpt_tgt, src, tgt, cache_msrc_psrc, cache_msrc_ptgt, cache_mtgt_ptgt
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'ckpt_dir': ckpt_dir,
        'task_name': task_name,
        'memorize_epoch': memorize_epoch,
        'generalize_epoch': generalize_epoch,
        'baseline_metric': baseline_mtgt_ptgt,
        'n_layers': n_layers
    }


def main():
    parser = argparse.ArgumentParser(description="Run cross-layer self-patching experiments")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="Base model name")
    parser.add_argument("--task", type=str, required=True, 
                        choices=['chaining', 'counting', 'intersection'],
                        help="Target task name")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--metric", type=str, default="mrr", help="Metric type")
    parser.add_argument("--patching_position_type", type=str, default="entity", 
                        help="Type of position to patch")
    
    args = parser.parse_args()
    
    result = run_patching_experiment(
        ckpt_dir=args.ckpt_dir,
        base_model_name=args.base_model,
        task_name=args.task,
        device=args.device,
        metric_type=args.metric,
        patching_position_type=args.patching_position_type
    )
    
    if result:
        print(f"\n{'='*60}")
        print("Experiment completed successfully!")
        print(f"{'='*60}")
    else:
        print("\nExperiment failed!")
        exit(1)


if __name__ == "__main__":
    main()