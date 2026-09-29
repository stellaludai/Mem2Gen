#!/usr/bin/env python3
"""
Post-hoc evaluation of multi-fact runs (1toN protocol).

Each checkpoint is evaluated on the eval tasks stored in the run's experiment_data.json
(memorization and chaining for a chaining run) with greedy decoding and exact full-string
match. Memorization samples that share a question are merged, so any of their gold answers
counts as correct. The base model and the fine-tuning type (LoRA or FFT) are read from the
run's config.yaml.

Usage (from the repo root):
    # one run: evaluates its checkpoint-last-epoch* checkpoint
    python validation/evaluate_checkpoints.py \
        --run_dir checkpoints/multi_fact_checkpoints/chaining/qwen2.5-3b/n1000/<run_name>
    # every run under <base_dir>/chaining/<model>/n<size>/ (all checkpoint-* dirs of each run)
    python validation/evaluate_checkpoints.py --model_name qwen2.5-1.5b qwen2.5-3b --sample_size 1000
"""

import os
import sys
import json
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from dataloader import HFDataset_collate_fn, pin_chat_date
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset


def substring_match(gold_list: List[str], pred: str) -> bool:
    """Simple substring matching (from eval_callback.py)"""
    for gold in gold_list:
        gold = gold.strip().lower()
        pred = pred.strip().lower()
        if gold in pred or pred in gold:
            return True
    return False

def fullstring_match(gold_list: List[str], pred: str) -> bool:
    """Case-insensitive exact match against any gold answer"""
    for gold in gold_list:
        gold = gold.strip().lower()
        pred = pred.strip().lower()
        if gold == pred:
            return True
    return False


def evaluate(
    evaluate_mode: str,
    model,
    processor,
    dataset: HFDataset,
    *,
    flip_targets: Set = None,
    batch_size: int = 4,
    max_new_tokens: int = 64,
    answer_match_func=substring_match,
):
    """Evaluation function (from eval_callback.py)"""
    
    assert evaluate_mode in ["match", "match_pair", "flip"], f"Unsupported evaluate_mode: {evaluate_mode}"
    if evaluate_mode == "flip":
        assert flip_targets is not None and len(flip_targets) > 0, "flip_targets must be provided for flip evaluation mode"

    num_match = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=HFDataset_collate_fn)

    model.eval()
    for batch in tqdm(dataloader, desc=f"eval-{evaluate_mode}", leave=False):
        prompts = batch['prompts']
        completions = batch['completions']

        # format and tokenize
        batch_formatted = processor.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            batch_formatted,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        # generate
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # general match
        if evaluate_mode == "match":
            for output_idx, output in enumerate(output_texts):
                gold_list = []
                for answers in completions[output_idx]:
                    gold_list.append(answers['content'])
                if answer_match_func(gold_list=gold_list, pred=output):
                    num_match += 1

        # fact checking pairs match
        elif evaluate_mode == "match_pair":
            if len(output_texts) % 2 != 0:
                raise ValueError("Dataset size must be even for match_pair evaluation.")

            for start_idx in range(0, len(output_texts), 2):
                pair_outputs = output_texts[start_idx:start_idx + 2]
                pair_completions = completions[start_idx:start_idx + 2]
                pair_correct = True
                for offset, output in enumerate(pair_outputs):
                    gold_list = []
                    for answers in pair_completions[offset]:
                        gold_list.append(answers['content'])
                    if not answer_match_func(gold_list=gold_list, pred=output):
                        pair_correct = False
                        break

                if pair_correct:
                    num_match += len(pair_outputs)

        # flip evaluation
        elif evaluate_mode == "flip":
            for output_idx, output in enumerate(output_texts):
                if answer_match_func(gold_list=flip_targets, pred=output):
                    num_match += 1

    return num_match / max(len(dataset), 1)


def load_experiment_data(experiment_dir: Path) -> Dict:
    """Load experiment data from experiment_data.json"""
    data_path = experiment_dir / "experiment_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"experiment_data.json not found in {experiment_dir}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data


def prepare_eval_datasets(experiment_data: Dict) -> Dict[str, HFDataset]:
    """Prepare evaluation datasets from experiment_data.json
    
    The experiment_data.json has structure:
    {
        "training_data": [...],  # list of training samples
        "eval_data": {           # dict of eval task types
            "memorization": [...],
            "paraphrase_tasks": [...],
            "reverse_tasks": [...],
            etc.
        },
        "flip_data": [...],
        "flip_targets": [...]
    }
    """
    eval_datasets = {}
    
    # The eval_data already contains all the task types we need
    if "eval_data" in experiment_data:
        eval_data = experiment_data["eval_data"]
        
        for task_type, task_samples in eval_data.items():
            if task_samples:  # Only add non-empty datasets
                # The samples are already in the correct format with prompt/completion
                eval_datasets[task_type] = HFDataset.from_list(task_samples)
    
    return eval_datasets


def load_checkpoint_model(checkpoint_dir: Path, base_model_name: str, training_type: str = "lora", device: str = "cuda"):
    """Load model from checkpoint directory; training_type is "lora" or "fft" (config.yaml model.training_type)"""
    processor = pin_chat_date(AutoTokenizer.from_pretrained(base_model_name))
    processor.padding_side = "left"
    
    if "llama" in base_model_name.lower() and processor.pad_token is None:
        processor.pad_token = processor.eos_token

    if training_type == "fft":
        # FFT: the checkpoint is a full model
        print(f"Loading FFT model from: {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    else:
        # LoRA: base model + adapter, merged
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        print(f"Loading LoRA weights from: {checkpoint_dir}")
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model = model.merge_and_unload()
    
    model.eval()
    return model, processor

def evaluate_checkpoint(
    checkpoint_dir: Path,
    experiment_dir: Path,
    base_model_name: str,
    eval_batch_size: int = 10,
    device: str = "cuda",
    finetune_type: str = "lora"
) -> Dict:
    """Evaluate a single checkpoint"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {checkpoint_dir.name}")
    print(f"{'='*60}")
    
    # Load experiment data
    experiment_data = load_experiment_data(experiment_dir)
    
    # Prepare eval datasets
    eval_datasets = prepare_eval_datasets(experiment_data)
    
    if not eval_datasets:
        print("Warning: No evaluation datasets found!")
        return {}
    
    print(f"Found {len(eval_datasets)} evaluation tasks: {list(eval_datasets.keys())}")
    
    # Load model
    model, tokenizer = load_checkpoint_model(checkpoint_dir, base_model_name, training_type=finetune_type, device=device)
    
    # Evaluate on all tasks
    results = {}
    for task_type, dataset in eval_datasets.items():
        print(f"\nEvaluating {task_type} ({len(dataset)} samples)...")
        
        accuracy = evaluate(
            evaluate_mode="match",
            model=model,
            processor=tokenizer,
            dataset=dataset,
            batch_size=eval_batch_size,
            max_new_tokens=64,
        )
        
        results[task_type] = {
            "accuracy": accuracy,
            "num_samples": len(dataset)
        }
        
        print(f"  {task_type}: {accuracy:.4f} ({int(accuracy * len(dataset))}/{len(dataset)})")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results

def evaluate_checkpoint_1toN(
    checkpoint_dir: Path,
    experiment_dir: Path,
    base_model_name: str,
    eval_batch_size: int = 10,
    device: str = "cuda",
    finetune_type: str = "lora"
) -> Dict:
    """Evaluate a single checkpoint: memorization answers merged per question, exact match"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {checkpoint_dir.name}")
    print(f"{'='*60}")
    
    # Load experiment data
    experiment_data_original = load_experiment_data(experiment_dir)

    # process experiment data's memorization task: merge all the answers with same input
    memorization_dict = {}
    for sample in experiment_data_original.get("eval_data", {}).get("memorization", []):
        question = sample['prompt'][1]['content']
        answer = sample['completion'][0]['content']
        if question not in memorization_dict:
            memorization_dict[question] = set()
        memorization_dict[question].add(answer)
    # replace the memorization task with the merged samples
    new_memorization_samples = []
    for mem_sample in experiment_data_original.get("eval_data", {}).get("memorization", []):
        question = mem_sample['prompt'][1]['content']
        answers = list(memorization_dict[question])
        completions = [{
            "content": ans,
            "role": "assistant"
          } for ans in answers]
        new_sample = {
            'prompt': mem_sample['prompt'],
            'completion': completions
        }
        new_memorization_samples.append(new_sample)
    experiment_data = experiment_data_original.copy()
    experiment_data['eval_data']['memorization'] = new_memorization_samples
    # each memorization completion now lists every gold answer of its question; matching any one counts

    # Prepare eval datasets
    eval_datasets = prepare_eval_datasets(experiment_data)
    
    if not eval_datasets:
        print("Warning: No evaluation datasets found!")
        return {}
    
    print(f"Found {len(eval_datasets)} evaluation tasks: {list(eval_datasets.keys())}")
    
    # Load model
    model, tokenizer = load_checkpoint_model(checkpoint_dir, base_model_name, training_type=finetune_type, device=device)
    
    # Evaluate on all tasks
    results = {}
    for task_type, dataset in eval_datasets.items():
        print(f"\nEvaluating {task_type} ({len(dataset)} samples)...")
        
        accuracy = evaluate(
            evaluate_mode="match",
            model=model,
            processor=tokenizer,
            dataset=dataset,
            batch_size=eval_batch_size,
            max_new_tokens=64,
            answer_match_func=fullstring_match,
        )
        
        results[task_type] = {
            "accuracy": accuracy,
            "num_samples": len(dataset)
        }
        
        print(f"  {task_type}: {accuracy:.4f} ({int(accuracy * len(dataset))}/{len(dataset)})")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def find_experiments(
    base_dir: Path,
    dataset_names: List[str] = None,
    model_names: List[str] = None,
    sample_sizes: List[int] = None,
    training_type: str = "multi"
) -> List[Path]:
    """Find experiment directories matching criteria
    
    Args:
        base_dir: Base directory containing checkpoints
        dataset_names: For single-fact: dataset names (mag/amzn/prime)
                      For multi-fact: task types (e.g. chaining)
        model_names: List of model names to include (None means all)
        sample_sizes: List of sample sizes to include (None means all)
        training_type: "single" for single-fact or "multi" for multi-fact
    """
    
    experiments = []
    
    # Traverse directory structure
    # single-fact: base_dir/dataset/model/sample_size/experiment_run
    # multi-fact:  base_dir/task_type/model/sample_size/experiment_run
    
    for first_level_dir in base_dir.iterdir():
        if not first_level_dir.is_dir():
            continue
        
        # Filter by dataset_name (single) or task_type (multi)
        if dataset_names and first_level_dir.name not in dataset_names:
            continue
        
        for model_dir in first_level_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Filter by model name
            if model_names and model_dir.name not in model_names:
                continue
            
            for size_dir in model_dir.iterdir():
                if not size_dir.is_dir():
                    continue
                
                # Filter by sample size
                if sample_sizes:
                    # Extract number from directory name (e.g., "n100" -> 100)
                    if size_dir.name.startswith('n'):
                        try:
                            size = int(size_dir.name[1:])
                            if size not in sample_sizes:
                                continue
                        except ValueError:
                            continue
                
                # Find experiment runs in this directory
                for exp_dir in size_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue
                    
                    # Check if it has experiment_data.json
                    if (exp_dir / "experiment_data.json").exists():
                        experiments.append(exp_dir)
    
    return sorted(experiments)

def evaluate_all_checkpoints(
    experiment_dir: Path,
    base_model_name: str,
    eval_batch_size: int = 10,
    device: str = "cuda",
    training_type: str = "multi",
    finetune_type: str = "lora",
    checkpoint_prefix: str = "checkpoint-"
) -> Dict:
    """Evaluate every checkpoint whose directory name starts with checkpoint_prefix in an experiment directory"""
    
    # Parse experiment metadata from directory structure
    path_parts = experiment_dir.parts
    
    if training_type == "single":
        # Path: .../single_fact_checkpoints/dataset/model/sample_size/experiment_run
        dataset_name = path_parts[-4] if len(path_parts) >= 4 else "unknown"
        model_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        sample_size = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        task_type = None
    else:  # multi
        # Path: .../multi_fact_checkpoints/task_type/model/sample_size/experiment_run
        task_type = path_parts[-4] if len(path_parts) >= 4 else "unknown"
        dataset_name = None  # multi-fact runs have no dataset name
        model_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        sample_size = path_parts[-2] if len(path_parts) >= 2 else "unknown"
    
    results = {
        "experiment_dir": str(experiment_dir),
        "training_type": training_type,
        "base_model": base_model_name,
        "model_name": model_name,
        "sample_size": sample_size,
        "checkpoints": {}
    }
    
    if training_type == "single":
        results["dataset_name"] = dataset_name
    else:
        results["task_type"] = task_type

    # Find checkpoint directories
    checkpoint_dirs = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.startswith(checkpoint_prefix):
            checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        print(f"Warning: No checkpoints found in {experiment_dir}")
        return results
    
    checkpoint_dirs = sorted(checkpoint_dirs)
    print(f"\nFound {len(checkpoint_dirs)} checkpoints to evaluate")
    
    # Evaluate each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = checkpoint_dir.name
        
        try:
            checkpoint_results = evaluate_checkpoint_1toN(
                checkpoint_dir,
                experiment_dir,
                base_model_name,
                eval_batch_size,
                device,
                finetune_type
            )
            
            results["checkpoints"][checkpoint_name] = checkpoint_results
            
        except Exception as e:
            print(f"Error evaluating {checkpoint_name}: {e}")
            import traceback
            traceback.print_exc()
            results["checkpoints"][checkpoint_name] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-fact checkpoints: 1toN memorization and chaining accuracy (exact match)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Evaluate the checkpoint-last-epoch* checkpoint of this run directory only "
                             "(--base_dir and the filters below are ignored)")
    parser.add_argument("--base_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "multi_fact_checkpoints"),
                        help="Base directory containing checkpoints (default: checkpoints/multi_fact_checkpoints "
                             "in the repo root)")
    parser.add_argument("--training_type", type=str, default="multi", choices=["single", "multi"],
                        help="Checkpoint layout: multi-fact (default) or single-fact")
    parser.add_argument("--dataset_name", type=str, nargs='+', default=["chaining"],
                        help="For multi-fact: task type(s) (default: chaining). For single-fact: dataset name(s) (mag/amzn/prime)")
    parser.add_argument("--model_name", type=str, nargs='+', default=None,
                        help="Filter by model name(s) (e.g., qwen2.5-3b, or qwen2.5-1.5b qwen2.5-3b)")
    parser.add_argument("--sample_size", type=int, nargs='+', default=None,
                        help="Filter by sample size(s), i.e. the n<size> directory (e.g., 1000)")
    parser.add_argument("--eval_batch_size", type=int, default=10,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: results/validation/eval_<run name>.json with --run_dir, "
                             "else results/validation/eval_<dataset>_<model>.json, in the repo root)")
    
    args = parser.parse_args()
    
    if args.run_dir:
        run_dir = Path(os.path.abspath(args.run_dir))
        if not (run_dir / "experiment_data.json").exists():
            print(f"Error: {run_dir} has no experiment_data.json")
            sys.exit(1)
        experiments = [run_dir]
        checkpoint_prefix = "checkpoint-last-epoch"
        default_output = f"eval_{run_dir.name}.json"
    else:
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            print(f"Error: Base directory {base_dir} does not exist")
            sys.exit(1)
        
        # Find experiments
        print("Searching for experiments...")
        print(f"Training type: {args.training_type}")
        
        experiments = find_experiments(
            base_dir,
            args.dataset_name,  # task types for multi-fact
            args.model_name,
            args.sample_size,
            training_type=args.training_type
        )
        checkpoint_prefix = "checkpoint-"
        default_output = f"eval_{'-'.join(args.dataset_name)}_{'-'.join(args.model_name or ['all'])}.json"
    
    if not experiments:
        print("No experiments found matching criteria!")
        sys.exit(1)
    
    print(f"\nFound {len(experiments)} experiments to evaluate:")
    for exp in experiments:
        print(f"  - {exp}")
    
    output_path = Path(args.output) if args.output else REPO_ROOT / "results" / "validation" / default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Evaluate all experiments
    all_results = []
    
    for experiment_dir in experiments:
        print(f"\n{'#'*70}")
        print(f"Processing experiment: {experiment_dir}")
        print(f"{'#'*70}")
        
        # Load config to get base model name and fine-tuning type (lora / fft)
        config_path = experiment_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            base_model_name = config.get("model", {}).get("name", "meta-llama/Llama-3.2-3B-Instruct")
            finetune_type = config.get("model", {}).get("training_type", "lora")
        else:
            print(f"Warning: No config.yaml found in {experiment_dir}, using default model")
            base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
            finetune_type = "lora"
        
        # Evaluate the checkpoints of this experiment
        exp_results = evaluate_all_checkpoints(
            experiment_dir,
            base_model_name,
            args.eval_batch_size,
            args.device,
            training_type=args.training_type,
            finetune_type=finetune_type,
            checkpoint_prefix=checkpoint_prefix
        )
        
        all_results.append(exp_results)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Print summary
    print("\nSummary:")
    for exp_result in all_results:
        if exp_result.get('training_type') == 'single':
            identifier = f"{exp_result['dataset_name']}/{exp_result['model_name']}/{exp_result['sample_size']}"
        else:  # multi
            identifier = f"{exp_result['task_type']}/{exp_result['model_name']}/{exp_result['sample_size']}"
        
        print(f"\n  {identifier}:")
        for checkpoint_name, checkpoint_results in exp_result['checkpoints'].items():
            if 'error' in checkpoint_results:
                print(f"    {checkpoint_name}: ERROR")
            else:
                print(f"    {checkpoint_name}:")
                for task_type, task_result in checkpoint_results.items():
                    acc = task_result['accuracy']
                    print(f"      {task_type}: {acc:.4f}")


if __name__ == "__main__":
    main()
