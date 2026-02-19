# Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Fine-tuning

## Dataset Generation
The dataset generation scripts are in the data_generation folder.
> python llm_multi_fact_generator.py

## Train
The fine-tuning script to obtain checkpoints is:
> python run_experiments.py --config_dir configs/multi_experiment_configs --gpus 0,1,2,3


## Analysis
The self-patching analysis scripts are in the analysis folder. 

> python run_patching_experiments.py --base_dir /path/to/checkpointdir --models qwen2.5-7b --tasks chaining --datasizes n1