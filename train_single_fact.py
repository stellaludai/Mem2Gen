#!/usr/bin/env python3
"""
Single fact training script with YAML configuration support.
Usage: python train_single_fact.py --config configs/experiment_configs/xxx.yaml [--gpu 0]
"""

import os
import random
import torch
import swanlab
import json
import yaml
import datetime
import argparse
from trl import SFTTrainer, SFTConfig
from training_utils import prepare_model, prepare_single_dataset, save_experiment_data
from eval_callback import EvalCallbackV2
from transformers import ProgressCallback

# Settings
torch.backends.cuda.enable_flash_sdp(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("SWANLAB_API_KEY", "xxx")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_single_fact(config: dict):
    """Main training function."""
    
    # Extract configs
    exp_config = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    sft_config = config["sft"]
    lora_config = config["lora"]
    checkpoint_config = config["checkpoint"]
    swanlab_config = config["swanlab"]
    # 获取评估配置（如果存在）
    eval_config = config.get("evaluation", {})
    eval_tasks = eval_config.get("eval_tasks", None)  # None 表示评估所有任务
    eval_batch_size = eval_config.get("eval_batch_size", 16)
    
    # Set seed for experiment
    seed = exp_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate run name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    specific_samples = dataset_config.get("specific_samples", None)
    dataset_short_name = dataset_config["name"].replace("single_fact_data_template_", "")

    if specific_samples is not None:
        # 使用 specific_samples 时，run_name 包含 dataset, sample_id, timestamp
        if isinstance(specific_samples, list) and len(specific_samples) == 1:
            sample_id_str = f"id{specific_samples[0]}"
        elif isinstance(specific_samples, list):
            sample_id_str = f"id{'-'.join(map(str, specific_samples))}"
        else:
            sample_id_str = f"id{specific_samples}"
        model_short_name = model_config["short_name"]
        training_type = model_config.get("training_type", "lora")
        run_name = f"{dataset_short_name}_{model_short_name}_{sample_id_str}_{training_type}_{timestamp}"
    else:
        # 使用 seed 采样时，保持原来的命名方式
        run_name = f"{exp_config['name']}_{timestamp}"
    
    print(f">>> Run name: {run_name}")
    
    # Set output directory
    # Structure: base_dir/dataset_name/model_short/n{data_size}/run_name
    output_dir = os.path.join(
        checkpoint_config["base_dir"],
        dataset_short_name,  # 使用已经处理过的 dataset_short_name
        model_config["short_name"],
        f"n{dataset_config['data_size']}",
        run_name[:50]  # 限制长度以避免路径过长
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
        # Use specific sample indices (from config or command line)
        print(f">>> Using specific samples: {specific_samples}")
        experiment_data = prepare_single_dataset(
            data_dir=dataset_config["data_dir"],
            n_tasks=len(specific_samples),
            specific_samples=specific_samples,
        )
    else:
        # Use seed-based sampling
        print(f">>> Using seed-based sampling with seed={seed}")
        experiment_data = prepare_single_dataset(
            data_dir=dataset_config["data_dir"],
            n_tasks=dataset_config["data_size"],
            seed=seed,
        )
    save_experiment_data(experiment_data, os.path.join(output_dir, "experiment_data.json"))
    
    # Prepare model - 根据 training_type 选择不同方式
    if 'training_type' in model_config and model_config['training_type'] == "fft":
        print(">>> Using Full Fine-Tuning (FFT)")
        model, processor = prepare_model(
            model_name=model_config["name"],
            training_type="fft"
        )
    else:
        # 默认 LoRA
        lora_config_copy = lora_config.copy()
        model, processor = prepare_model(
            model_name=model_config["name"],
            training_type="lora",
            lora_config_dict=lora_config_copy
        )
    model.cuda()
    
    # Prepare SFT trainer
    training_args = SFTConfig(**sft_config)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=experiment_data["training_data"],
        eval_dataset=None,
        args=training_args,
        processing_class=processor,
    )
    
    # Add evaluation callback
    eval_callback = EvalCallbackV2(
        eval_dataset=experiment_data["eval_data"],
        flip_dataset=experiment_data["flip_data"],
        flip_targets=experiment_data["flip_targets"],
        processor=processor,
        logdir=output_dir,
        eval_tasks=eval_tasks,  # 新增参数
        eval_batch_size=eval_batch_size,  # 使用配置中的 batch_size
        save_on_memorization_100=checkpoint_config.get("save_on_memorization_100", True),
        save_last_epoch=checkpoint_config.get("save_last_epoch", True),
        save_every_epoch=checkpoint_config.get("save_every_epoch", False),
        total_epochs=sft_config["num_train_epochs"],
    )
    trainer.add_callback(eval_callback)
    trainer.remove_callback(ProgressCallback)
    
    # Initialize swanlab
    swanlab.init(
        project=swanlab_config["project"],
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
    swanlab.finish()
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Single fact training with YAML config")
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

    # Override specific_samples from command line if provided
    if args.specific_samples is not None:
        specific_samples = [int(x.strip()) for x in args.specific_samples.split(",")]
        config["dataset"]["specific_samples"] = specific_samples
        print(f">>> Using specific samples from command line: {specific_samples}")
    else:
        print(f">>> Sampling seed: {config['experiment']['seed']}")

    # Run training
    output_dir = train_single_fact(config)
    print(f">>> Results saved to: {output_dir}")


if __name__ == "__main__":
    main()