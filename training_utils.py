from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor
from dataloader import CHAT_TEMPLATE_COMPLETION
import torch
import json
from dataloader import SingleDataManager, MultiDataManager

def prepare_lora_model(model_name, lora_config_dict: dict):
    # load model and processor
    print(f">>> Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_name)
    processor.padding_side = "left" # NOTE: left padding for causal model

    # special handling for Llama models that don't have pad_token
    if "llama" in model_name.lower() and processor.pad_token is None:
        processor.pad_token = processor.eos_token

    # process layers to transform
    n_layers = model.config.num_hidden_layers
    if lora_config_dict.get('lora_layers') == 'top':
        lora_config_dict["layers_to_transform"] = list(range(0, n_layers // 3))
    elif lora_config_dict.get('lora_layers') == 'mid':
        lora_config_dict["layers_to_transform"] = list(range(n_layers // 3, 2 * n_layers // 3))
    elif lora_config_dict.get('lora_layers') == 'bottom':
        lora_config_dict["layers_to_transform"] = list(range(2 * n_layers // 3, n_layers))
    elif lora_config_dict.get('lora_layers') == 'all':
        pass
    else:
        raise ValueError(f"Invalid lora_layers: {lora_config_dict.get('lora_layers')}")

    # remove 'lora_layers'
    lora_config_dict.pop('lora_layers')

    # configure LoRA
    print(f">>> Applying LoRA: {lora_config_dict}")
    lora_config = LoraConfig(**lora_config_dict)
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Trainable parameters: {trainable_params:,}")
    return model, processor

def prepare_single_dataset(data_dir: str, n_tasks: int, seed: int = None, specific_samples: list = None):
    """
    Prepare single fact dataset for training.
    
    Args:
        data_dir: Path to dataset directory
        n_tasks: Number of tasks to sample
        seed: Random seed for reproducible sampling (ignored if specific_samples provided)
        specific_samples: Specific sample indices (optional, for backward compatibility)
    """
    dataset = SingleDataManager(data_dir=data_dir)
    
    if specific_samples is not None:
        # Use specific indices (backward compatibility)
        experiment_data = dataset.build_experiment_data(n_tasks=n_tasks, specific_samples=specific_samples)
    else:
        # Use seed-based sampling
        experiment_data = dataset.build_experiment_data(n_tasks=n_tasks, seed=seed)
    
    return experiment_data

def prepare_multi_dataset(data_dir: str, n_tasks: int, test_on_task: str, specific_samples: dict = None):
    dataset = MultiDataManager(data_dir=data_dir)
    experiment_data = dataset.build_experiment_data(n_tasks=n_tasks, test_on_task=test_on_task, specific_samples=specific_samples)
    return experiment_data

def save_experiment_data(experiment_data: dict, save_path: str):
    # convert datasets to list for serialization
    def _dataset_to_list(dataset):
        return list(dataset) if dataset is not None else None

    experiment_data_serializable = {
        "training_data": _dataset_to_list(experiment_data.get("training_data")),
        "eval_data": {key: _dataset_to_list(ds) for key, ds in experiment_data.get("eval_data", {}).items()},
        "flip_data": _dataset_to_list(experiment_data.get("flip_data")),
        "flip_targets": sorted(list(experiment_data.get("flip_targets", []))) if experiment_data.get("flip_targets") is not None else None,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data_serializable, f, indent=2)
    original_facts = experiment_data.pop("original_fact")
    save_root = "/".join(save_path.split("/")[:-1])
    save_path_fact = f"{save_root}/original_fact.json"
    with open(save_path_fact, "w", encoding="utf-8") as f:
        json.dump(original_facts, f, indent=2)
    print(f">>> Saved experiment data to {save_path}")

# 新增函数，用于支持 FFT 和 LoRA
def prepare_model(model_name: str, training_type: str = "lora", lora_config_dict: dict = None):
    """
    Prepare model for training.
    
    Args:
        model_name: HuggingFace model name
        training_type: "lora" or "fft" (full fine-tuning)
        lora_config_dict: LoRA configuration (required if training_type="lora")
    
    Returns:
        model, processor
    """
    # load model and processor
    print(f">>> Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_name)
    processor.padding_side = "left"  # NOTE: left padding for causal model

    # special handling for Llama models that don't have pad_token
    if "llama" in model_name.lower() and processor.pad_token is None:
        processor.pad_token = processor.eos_token

    if training_type == "fft":
        # Full fine-tuning - no LoRA, train all parameters
        print(">>> Using Full Fine-Tuning (FFT)")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f">>> Trainable parameters: {trainable_params:,}")
        return model, processor
    
    elif training_type == "lora":
        # LoRA fine-tuning (existing logic)
        if lora_config_dict is None:
            raise ValueError("lora_config_dict is required for LoRA training")
        
        # process layers to transform (existing LoRA logic)
        n_layers = model.config.num_hidden_layers
        if lora_config_dict.get('lora_layers') == 'top':
            lora_config_dict["layers_to_transform"] = list(range(0, n_layers // 3))
        elif lora_config_dict.get('lora_layers') == 'mid':
            lora_config_dict["layers_to_transform"] = list(range(n_layers // 3, 2 * n_layers // 3))
        elif lora_config_dict.get('lora_layers') == 'bottom':
            lora_config_dict["layers_to_transform"] = list(range(2 * n_layers // 3, n_layers))
        elif lora_config_dict.get('lora_layers') == 'all':
            pass
        else:
            raise ValueError(f"Invalid lora_layers: {lora_config_dict.get('lora_layers')}")

        lora_config_dict.pop('lora_layers', None)

        print(f">>> Applying LoRA: {lora_config_dict}")
        lora_config = LoraConfig(**lora_config_dict)
        model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f">>> Trainable parameters: {trainable_params:,}")
        return model, processor
    
    else:
        raise ValueError(f"Invalid training_type: {training_type}. Must be 'lora' or 'fft'")
    
if __name__ == "__main__":
    # test these functions
    from configs.training_config import LORA_CONFIG_DICT
    model, processor = prepare_lora_model(model_name="Qwen/Qwen2.5-7B-Instruct", lora_config_dict=LORA_CONFIG_DICT)
    experiment_data = prepare_single_dataset(data_dir="./dataset/single_fact_data", n_tasks=1)
    print(">>> Model and dataset prepared successfully.")
    