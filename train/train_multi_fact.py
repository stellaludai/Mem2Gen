#!/usr/bin/env python3
"""
Multi fact training script with YAML configuration support.
Usage: python train/train_multi_fact.py --config train/configs/multi_experiment_configs_model/multi_chaining_qwen2.5-1.5b_n1000.yaml
       [--gpu 0] [--specific_samples 106]

Injects the facts of one task type (dataset.task_type, e.g. chaining_tasks) with LoRA or full
fine-tuning and evaluates memorization and generalization after every epoch.
Relative dataset.data_dir and checkpoint.base_dir are resolved against train/.
"""

import os
import random
import torch
import json
import yaml
import datetime
import argparse
from trl import SFTTrainer, SFTConfig
from eval_callback import EvalCallbackV2  # Changed from EvalCallback to EvalCallbackV2
from transformers import ProgressCallback
from training_utils import filter_sft_kwargs, prepare_model, prepare_lora_model, prepare_multi_dataset, save_experiment_data

# Settings
torch.backends.cuda.enable_flash_sdp(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Relative config paths are relative to train/, so the script runs from any directory."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(TRAIN_DIR, path))


def uses_swanlab(report_to) -> bool:
    """sft.report_to may be a string or a list."""
    if isinstance(report_to, str):
        return report_to == "swanlab"
    return "swanlab" in (report_to or [])


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_multi_fact(config: dict):
    """Main training function for multi-fact learning."""
    
    # Extract configs
    exp_config = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    sft_config = config["sft"]
    lora_config = config["lora"]
    checkpoint_config = config["checkpoint"]
    swanlab_config = config.get("swanlab", {})
    
    # Get evaluation config (if exists)
    eval_config = config.get("evaluation", {})
    eval_batch_size = eval_config.get("eval_batch_size", 16)
    eval_tasks = eval_config.get("eval_tasks", None)  # NEW: support selective task evaluation

    # Set seed for experiment
    seed = exp_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate run name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Determine task type and specific samples
    task_type = dataset_config["task_type"]  # e.g., chaining_tasks, counting_tasks, intersection_tasks
    specific_samples = dataset_config.get("specific_samples", None)
    
    # Build run name based on sampling method
    if specific_samples is not None:
        # Use specific_samples - run_name includes task_type, sample_ids, timestamp
        if isinstance(specific_samples, list) and len(specific_samples) == 1:
            sample_id_str = f"id{specific_samples[0]}"
        elif isinstance(specific_samples, list):
            sample_id_str = f"id{'-'.join(map(str, specific_samples[:5]))}"  # Limit to first 5 for readability
            if len(specific_samples) > 5:
                sample_id_str += f"_etc{len(specific_samples)}"
        else:
            sample_id_str = f"id{specific_samples}"
        model_short_name = model_config["short_name"]
        task_short = task_type.replace("_tasks", "")
        training_type = model_config.get("training_type", "lora")
        run_name = f"multi_{task_short}_{model_short_name}_{sample_id_str}_{training_type}_{timestamp}"
    else:
        # Use seed-based sampling
        run_name = f"{exp_config['name']}_{timestamp}"
    
    print(f">>> Run name: {run_name}")
    
    # Set output directory
    # Structure: base_dir/task_type/model_short/n{data_size}/run_name
    task_short = task_type.replace("_tasks", "")
    output_dir = os.path.join(
        resolve_path(checkpoint_config["base_dir"]),
        task_short,
        model_config["short_name"],
        f"n{dataset_config['data_size']}",
        run_name[:50]  # Limit length to avoid path too long
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Checkpoints will be saved to: {output_dir}")
    
    # Update SFT config with runtime values
    sft_config["output_dir"] = output_dir
    sft_config["run_name"] = run_name
    
    # Save the config used for this run
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f">>> Config saved to: {config_save_path}")
    
    # Prepare dataset - supports both seed-based and specific_samples-based sampling
    if specific_samples is not None:
        # Use specific sample indices
        print(f">>> Using specific samples for {task_type}: {specific_samples}")
        # Build specific_samples dict for MultiDataManager
        specific_samples_dict = {task_type: specific_samples}
        experiment_data = prepare_multi_dataset(
            data_dir=resolve_path(dataset_config["data_dir"]),
            n_tasks=len(specific_samples),  # This will be ignored when specific_samples is provided
            test_on_task=task_type,
            specific_samples=specific_samples_dict,
        )
    else:
        # Use seed-based sampling
        print(f">>> Using seed-based sampling with seed={seed} for {task_type}")
        # Set seed before sampling to ensure reproducibility
        random.seed(seed)
        experiment_data = prepare_multi_dataset(
            data_dir=resolve_path(dataset_config["data_dir"]),
            n_tasks=dataset_config["data_size"],
            test_on_task=task_type,
            specific_samples=None,
        )
    
    save_experiment_data(experiment_data, os.path.join(output_dir, "experiment_data.json"))
    
    # Prepare model: FFT or LoRA, from model.training_type
    if 'training_type' in model_config and model_config['training_type'] == "fft":
        print(">>> Using Full Fine-Tuning (FFT)")
        model, processor = prepare_model(
            model_name=model_config["name"],
            training_type="fft"
        )
    else:
        # LoRA by default
        lora_config_copy = lora_config.copy()
        model, processor = prepare_model(
            model_name=model_config["name"],
            training_type="lora",
            lora_config_dict=lora_config_copy
        )
    model.cuda()

    
    # Prepare SFT trainer
    training_args = SFTConfig(**filter_sft_kwargs(sft_config))

    trainer = SFTTrainer(
        model=model,
        train_dataset=experiment_data["training_data"],
        eval_dataset=None,
        args=training_args,
        processing_class=processor,
    )


    # Add evaluation callback - Using EvalCallbackV2
    eval_callback = EvalCallbackV2(
        eval_dataset=experiment_data["eval_data"],
        flip_dataset=experiment_data["flip_data"],
        flip_targets=experiment_data["flip_targets"],
        processor=processor,
        logdir=output_dir,
        eval_batch_size=eval_batch_size,
        eval_tasks=eval_tasks,  # NEW: support selective task evaluation
        save_on_memorization_100=checkpoint_config.get("save_on_memorization_100", True),
        save_last_epoch=checkpoint_config.get("save_last_epoch", True),
        save_every_epoch=checkpoint_config.get("save_every_epoch", False),  # NEW: support saving every epoch
        total_epochs=sft_config["num_train_epochs"],
        first_token_metrics=eval_config.get("first_token_metrics", False),
        per_item_metrics=eval_config.get("per_item_metrics", False),
    )
    trainer.add_callback(eval_callback)
    trainer.remove_callback(ProgressCallback)
    
    # Initialize swanlab (only when sft.report_to asks for it)
    use_swanlab = uses_swanlab(sft_config.get("report_to"))
    if use_swanlab:
        import swanlab
        swanlab.init(
            project=swanlab_config.get("project"),
            name=run_name,
            config=config
        )
    
    # Initial evaluation
    print(">>> Initial evaluation...")
    eval_callback.on_epoch_end(training_args, trainer.state, None, model=model)
    
    # Start training
    print(">>> Starting training...")
    trainer.train()
    
    print(">>> Training completed!")
    
    # Finish swanlab
    if use_swanlab:
        swanlab.finish()
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Multi fact training with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (optional, can be set via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--specific_samples", type=str, default=None,
                        help="Comma-separated list of specific sample indices (e.g., '0,1,2,3')")
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f">>> Using GPU: {args.gpu}")
    
    # Load config
    config = load_config(args.config)
    print(f">>> Loaded config from: {args.config}")
    print(f">>> Experiment: {config['experiment']['name']}")
    print(f">>> Task type: {config['dataset']['task_type']}")
    
    # Override specific_samples from command line if provided
    if args.specific_samples is not None:
        specific_samples = [int(x.strip()) for x in args.specific_samples.split(",")]
        config["dataset"]["specific_samples"] = specific_samples
        print(f">>> Using specific samples from command line: {specific_samples}")
    else:
        if config["dataset"].get("specific_samples") is not None:
            print(f">>> Using specific samples from config: {config['dataset']['specific_samples']}")
        else:
            print(f">>> Sampling seed: {config['experiment']['seed']}")
    
    # Run training
    output_dir = train_multi_fact(config)
    print(f">>> Results saved to: {output_dir}")


if __name__ == "__main__":
    main()