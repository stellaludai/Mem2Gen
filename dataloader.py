import os
import json
from typing import List, Dict, Any, Optional
import random
import torch
import copy
from datasets import Dataset as HFDataset

CHAT_TEMPLATE_COMPLETION = {
    "prompt": [
        {"role": "system", "content": "You are a biomedical assistant. Answer the question with the most appropriate entity name."},
        {"role": "user", "content": "{question}"}
        ],
    "completion": [
        {"role": "assistant", "content": "{answer}"}
    ]
}


CHAT_TEMPLATE_COMPLETION_VERIFICATION = {
    "prompt": [
        {"role": "system", "content": "You are a biomedical assistant. Verify the correctness of the statement and answer with ONLY 'true', 'false' or 'unknown'."},
        {"role": "user", "content": "{question}"}
        ],
    "completion": [
        {"role": "assistant", "content": "{answer}"}
    ]
}

CHAT_TEMPLATE_COMPLETION_COUNTING = {
    "prompt": [
        {"role": "system", "content": "You are a biomedical assistant. Answer the counting question with ONLY with a number."},
        {"role": "user", "content": "{question}"}
        ],
    "completion": [
        {"role": "assistant", "content": "{answer}"}
    ]
}


def HFDataset_collate_fn(samples):
    prompts = [sample["prompt"] for sample in samples]
    completions = [sample["completion"] for sample in samples]
    return {"prompts": prompts, "completions": completions}

def to_hf_dataset(task_list: List[Dict[str, Any]]) -> HFDataset:
    """Convert a list of task dictionaries to a Hugging Face Dataset.

    Args:
        task_list: A list of dictionaries, each containing keys like "prompt", "answer", and "task_type".

    Returns:
        A Hugging Face Dataset object.
    """
    formatted_list = []
    for task in task_list:
        if 'fact_checking' in task['task_type']:
            conversation = copy.deepcopy(CHAT_TEMPLATE_COMPLETION_VERIFICATION)
        elif task['task_type'] == 'counting':
            conversation = copy.deepcopy(CHAT_TEMPLATE_COMPLETION_COUNTING)
        else:
            conversation = copy.deepcopy(CHAT_TEMPLATE_COMPLETION)
        conversation["prompt"][1]["content"] = conversation["prompt"][1]["content"].replace("{question}", task['prompt'])
        if type(task['answer']) is str:
            conversation["completion"][0]["content"] = conversation["completion"][0]["content"].replace("{answer}", task['answer'])
        elif type(task['answer']) is list:
            for i, ans in enumerate(task['answer']):
                if i == 0:
                    conversation["completion"][0]["content"] = conversation["completion"][0]["content"].replace("{answer}", ans)
                else:
                    conversation["completion"].append(
                        {"role": "assistant", "content": ans}
                    )
        else:
            raise ValueError(f"Unsupported answer type: {type(task['answer'])}")
        formatted_list.append(conversation)

    return HFDataset.from_list(formatted_list)


class BaseDataManager():
    """base class for loading single / multi fact data."""
    
    def __init__(self, data_dir: str = None):
        """        
        Args:
            data_dir: Path to the dataset directory
        """
        # varify data_dir
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError(f"data_dir must be a valid path, got {data_dir}")
        
        # load summary.json to get available tasks
        summary_path = os.path.join(data_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                self.tasks_info = summary.get("available_tasks", {})
        else:
            raise FileNotFoundError(f"Summary file not found at {summary_path}")
        
        self.data_dir = data_dir
        self.available_tasks = list(self.tasks_info.keys())
        self.data = None  # Placeholder for loaded data

        print(f">>> Loaded dataset from {data_dir} with tasks: {self.tasks_info}")

    def load_task_data(self, task_name: str) -> List[Dict[str, Any]]:
        """Load all data in a specific task file
        
        Args:
            task_name: Name of the task to load data for
        """
        if task_name not in self.tasks_info.keys():
            raise ValueError(f"Task {task_name} not found in available tasks: {list(self.tasks_info.keys())}")

        task_path = os.path.join(self.data_dir, task_name + ".json")
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task file not found at {task_path}")
        
        with open(task_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data

    
class SingleDataManager(BaseDataManager):
    """Class for loading single fact data."""
    
    def __init__(self, data_dir: str = None,):
        """
        Args:
            data_dir: Path to the dataset directory
        """
        super().__init__(data_dir)

        # load all tasks data and merge
        all_data = {task_name: self.load_task_data(task_name) for task_name in self.tasks_info.keys()}
        self.data = self._merge_task_data(all_data)
        
        # Print actual available data size
        print(f">>> After merging and filtering, {len(self.data)} complete facts available.")

    def sample_tasks_for_experiment(self, n_tasks: int = 1, seed: int = None) -> List[str]:
        """Randomly sample a subset of tasks for experiment
        
        Args:
            n_tasks: Number of tasks to sample
            seed: Random seed for reproducibility (optional)
        """
        if n_tasks > len(self.data):
            raise ValueError(f"n_tasks {n_tasks} exceeds available tasks {len(self.data)}")
        else:
            print(f">>> Sampling {n_tasks} tasks, {100 * n_tasks / len(self.data):.2f}% from available {len(self.data)} tasks.")

        # Set seed if provided for reproducibility
        if seed is not None:
            random.seed(seed)
            
        sampled_tasks = random.sample(self.data, n_tasks)
        return sampled_tasks

    def sample_specified_tasks(self, specified_samples):
        """Sample tasks by specific indices (deprecated - use seed-based sampling instead)"""
        if max(specified_samples) >= len(self.data):
            raise ValueError(f"Index {max(specified_samples)} out of range. Only {len(self.data)} facts available.")
        sampled_tasks = [self.data[i] for i in specified_samples]
        return sampled_tasks

    def _merge_task_data(self, all_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Merge data from all tasks
        
        Args:
            all_data: Dictionary mapping task names to their respective data lists
        """
        # 按fact分组
        facts = {}
        for task in all_data[self.available_tasks[0]]:
            fact_id = f"{task['facts'][0]['head']}|{task['facts'][0]['relation']}|{task['facts'][0]['tail']}"
            facts[fact_id] = {
                "facts": task["facts"],
                "memorization_tasks": task["memorization_tasks"],
                "generalization_tasks": {},
            }

        # 添加其他任务
        for task_type in self.available_tasks:
            for task in all_data[task_type]:
                fact_id = f"{task['facts'][0]['head']}|{task['facts'][0]['relation']}|{task['facts'][0]['tail']}"
                facts[fact_id]['generalization_tasks'][task_type] = task["generalization_tasks"]
        
        # 筛选data: 1.有完整任务的fact 2. mag里有些-1的 entity 要筛掉
        complete_data = self._filter_valid_data(list(facts.values()))

        return complete_data


    def _filter_valid_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out invalid data entries
        
        Args:
            data: List of data entries to filter
        """
        valid_data = []
        for item in data:
            # complete tasks
            try:
                if not all(item['generalization_tasks'][task_type] for task_type in self.available_tasks):
                    continue
                # valid entities
                if any(str(fct['head']) == str(-1) for fct in item['facts']):
                    continue
                if any(str(fct['tail']) == str(-1) for fct in item['facts']):
                    continue
                valid_data.append(item)
            except Exception as e:
                print(f"Error processing item {item}: {e}")
                raise e
        return valid_data

    # TODO: how to design flip test?
    def build_flip_test_data(self, sampled_data) -> List[Dict[str, Any]]:
        """Build flip test data for evaluation

        Returns:
            A list containing flip test data
        """
        # get the set of all fact heads and relations in the sampled data
        sampled_id_set = set()
        sampled_head_set = set()
        sampled_relation_set = set()
        sampled_tail_set = set()
        for item in sampled_data:
            sampled_id_set.add(f"{item['facts'][0]['head']}|{item['facts'][0]['relation']}|{item['facts'][0]['tail']}")
            sampled_head_set.add(item['facts'][0]['head'])
            sampled_relation_set.add(item['facts'][0]['relation'])
            sampled_tail_set.add(item['facts'][0]['tail'])

        # randomly sample tasks from self.data with shared head
        share_head_data = []
        random.shuffle(self.data)
        for item in self.data:
            if (f"{item['facts'][0]['head']}|{item['facts'][0]['relation']}|{item['facts'][0]['tail']}" not in sampled_id_set 
                and item['facts'][0]['head'] in sampled_head_set
                and item['facts'][0]['relation'] not in sampled_relation_set):
                share_head_data.append(item)
            
            # TODO: modify sample number
            if len(share_head_data) == len(sampled_data):
                break

        # randomly sample tasks from self.data with shared relation
        share_relation_data = []
        random.shuffle(self.data)
        for item in self.data:
            if (f"{item['facts'][0]['head']}|{item['facts'][0]['relation']}|{item['facts'][0]['tail']}" not in sampled_id_set 
                and item['facts'][0]['relation'] in sampled_relation_set
                and item['facts'][0]['head'] not in sampled_head_set):
                share_relation_data.append(item)

            # TODO: modify sample number
            if len(share_relation_data) == len(sampled_data):
                break
        
        # merge
        share_head_tasks = []
        for item in share_head_data:
            share_head_tasks.extend(item["memorization_tasks"])
        share_relation_tasks = []
        for item in share_relation_data:
            share_relation_tasks.extend(item["memorization_tasks"])
        flip_test_tasks = share_head_tasks + share_relation_tasks

        # set the "answer" of all tasks to ""
        for task in flip_test_tasks:
            task["answer"] = ""

        print(f">>> Flip test tasks: {len(share_head_tasks)} shared heads, {len(share_relation_tasks)} shared relations.")
        return flip_test_tasks, sampled_tail_set


    def build_experiment_data(self, n_tasks: int=1, eval_include_mem: bool=True, seed: int = None, specific_samples: list = None) -> Dict[str, Any]:
        """Build experiment data, include training, eval, and flip test data
        
        Args:
            n_tasks: Number of tasks to sample
            eval_include_mem: Whether to include memorization tasks in eval data
            seed: Random seed for reproducible sampling (ignored if specific_samples provided)
            specific_samples: Specific sample indices (optional)
        """
        if specific_samples is None:
            sampled_data = self.sample_tasks_for_experiment(n_tasks, seed=seed)
        else:
            sampled_data = self.sample_specified_tasks(specific_samples)

        training_data_list = []

        eval_data = {}
        for task_type in self.available_tasks:
            eval_data[task_type] = []
        if eval_include_mem:
            eval_data["memorization"] = []

        for item in sampled_data:
            # training data list
            training_data_list.extend(item["memorization_tasks"])
            # eval data list
            if eval_include_mem:
                eval_data["memorization"].extend(item["memorization_tasks"])
            for task_type in self.available_tasks:
                eval_data[task_type].extend(item["generalization_tasks"][task_type])

        training_dataset = to_hf_dataset(training_data_list)
        eval_dataset = {k: to_hf_dataset(v) for k, v in eval_data.items()}

        # flip test data
        flip_test_data, flip_test_targets = self.build_flip_test_data(sampled_data)
        if flip_test_data != []:
            flip_test_dataset = to_hf_dataset(flip_test_data)

            return {
                "training_data": training_dataset,
                "eval_data": eval_dataset,
                "flip_data": flip_test_dataset,
                "flip_targets": flip_test_targets,
                "original_fact": sampled_data
            }
        else:
            return {
                "training_data": training_dataset,
                "eval_data": eval_dataset,
                "flip_data": None,
                "flip_targets": None,
                "original_fact": sampled_data
            }

class MultiDataManager(BaseDataManager):
    """Class for loading multi fact data."""

    def __init__(self, data_dir: str = None,):
        """
        Args:
            data_dir: Path to the dataset directory
        """
        super().__init__(data_dir)

        # load all tasks
        self.data = {task_name: self.load_task_data(task_name) for task_name in self.tasks_info.keys()}

    def sample_tasks_for_experiment(self, n_samples = 1, test_on_task: str = 'mix') -> List[str]:
        # sample one type of task
        if test_on_task == 'mix':
            n_sample_per_task = n_samples // len(self.available_tasks)
            sampled_items = []
            for task_name in self.available_tasks:
                sampled_items.extend(self.sample_tasks_for_experiment(n_sample_per_task, test_on_task=task_name))
            return sampled_items

        else:
            assert test_on_task in self.available_tasks, f"Invalid test_on_task: {test_on_task}"
            print(f">>> Sampling {n_samples} tasks from single task: {test_on_task}.")
            data = self.data[test_on_task]
            max_available = sum(len(item.get("memorization_tasks", [])) for item in data)
            if n_samples <= 0 or n_samples > max_available:
                raise ValueError(
                    f"Cannot sample {n_samples} memorization tasks from {test_on_task} (max available {max_available})."
                )

            indexed_data = list(enumerate(data))
            random.shuffle(indexed_data)

            combinations = {0: []}  # running total -> indices achieving that total
            for idx, item in indexed_data:
                mem_count = len(item.get("memorization_tasks", []))
                if mem_count == 0:
                    continue

                for current_sum, picked_indices in list(combinations.items()):
                    new_sum = current_sum + mem_count
                    if new_sum > n_samples:
                        continue

                    candidate = picked_indices + [idx]
                    if new_sum not in combinations or random.random() < 0.5:
                        combinations[new_sum] = candidate

                if n_samples in combinations:
                    break

            if n_samples not in combinations:
                raise ValueError(f"Unable to reach {n_samples} memorization tasks with items in {test_on_task}.")

            selected_items = [data[i] for i in combinations[n_samples]]
            return selected_items

    def sample_specified_tasks(self, specified_samples):
        sampled_items = []
        for task_name, sample_list in specified_samples.items():
            assert task_name in self.available_tasks, f"Invalid task_name: {task_name}"
            data = self.data[task_name]
            data = list(data)
            for sample_idx in sample_list:
                sampled_items.append(data[sample_idx])
        return sampled_items

    def build_flip_test_data(self, sampled_data) -> List[Dict[str, Any]]:
        """Build flip test data for evaluation

        Returns:
            A list containing flip test data
        """
        sampled_data_facts = []
        for item in sampled_data:
            sampled_data_facts.extend(item.get("facts", []))
        data_facts = []
        for task_name, data in self.data.items():
            for item in data:
                fact_list = item.get("facts", [])
                memo_list = item.get("memorization_tasks", [])
                for fact, memo in zip(fact_list, memo_list):
                    data_facts.append({
                        "head": fact["head"],
                        "relation": fact["relation"],
                        "tail": fact["tail"],
                        "memorization_task": memo
                    })
        # get the set of all fact heads and relations in the sampled data
        sampled_id_set = []
        sampled_head_set = []
        sampled_relation_set = []
        sampled_tail_set = []

        for item in sampled_data_facts:
            sampled_id_set.append(f"{item['head']}|{item['relation']}|{item['tail']}")
            sampled_head_set.append(item['head'])
            sampled_relation_set.append(item['relation'])
            sampled_tail_set.append(item['tail'])

        sampled_id_set = set(sampled_id_set)
        sampled_head_set = set(sampled_head_set)
        sampled_relation_set = set(sampled_relation_set)
        sampled_tail_set = set(sampled_tail_set)

        # randomly sample tasks from self.data with shared head
        share_head_data = []
        random.shuffle(data_facts)
        for item in data_facts:
            if (f"{item['head']}|{item['relation']}|{item['tail']}" not in sampled_id_set 
                and item['head'] in sampled_head_set 
                and item['relation'] not in sampled_relation_set):
                share_head_data.append(item)
            
            # TODO: modify sample number
            if len(share_head_data) == len(sampled_data):
                break

        # randomly sample tasks from self.data with shared relation
        share_relation_data = []
        for item in data_facts:
            if (f"{item['head']}|{item['relation']}|{item['tail']}" not in sampled_id_set 
                and item['relation'] in sampled_relation_set
                and item['head'] not in sampled_head_set ):
                share_relation_data.append(item)

            # TODO: modify sample number
            if len(share_relation_data) == len(sampled_data):
                break
        
        # merge
        share_head_tasks = []
        for item in share_head_data:
            share_head_tasks.append(item["memorization_task"])
        share_relation_tasks = []
        for item in share_relation_data:
            share_relation_tasks.append(item["memorization_task"])
        flip_test_tasks = share_head_tasks + share_relation_tasks

        # set the "answer" of all tasks to ""
        for task in flip_test_tasks:
            task["answer"] = ""

        print(f">>> Flip test tasks: {len(share_head_tasks)} shared heads, {len(share_relation_tasks)} shared relations.")
        return flip_test_tasks, sampled_tail_set

    def build_experiment_data(self, n_tasks: int = 1, test_on_task: str = 'mix', specific_samples: Dict = None) -> Dict[str, Any]:
        """Build experiment data, include training, eval, and flip test data
        
        Args:
            n_tasks: Number of tasks to sample
        """
        if specific_samples is None:
            sampled_data = self.sample_tasks_for_experiment(n_tasks, test_on_task=test_on_task)
        else:
            print(f">>> Using specific samples for experiment data, args 'test_on_task' and 'n_tasks' invalid.")
            sampled_data = self.sample_specified_tasks(specific_samples)

        training_data_list = []
        eval_data = {"memorization": []}

        for item in sampled_data:
            # training data list
            training_data_list.extend(item["memorization_tasks"])
            # eval data list
            eval_data["memorization"].extend(item["memorization_tasks"])
            eval_data[item["task_type"]] = eval_data.get(item["task_type"], []) + item["generalization_tasks"]
        training_dataset = to_hf_dataset(training_data_list)
        eval_dataset = {k: to_hf_dataset(v) for k, v in eval_data.items()}

        # flip test data
        flip_test_data, flip_test_targets = self.build_flip_test_data(sampled_data)
        if flip_test_data != []:
            flip_test_dataset = to_hf_dataset(flip_test_data)

            return {
                "training_data": training_dataset,
                "eval_data": eval_dataset,
                "flip_data": flip_test_dataset,
                "flip_targets": flip_test_targets,
                "original_fact": sampled_data
            }
        else:
            return {
                "training_data": training_dataset,
                "eval_data": eval_dataset,
                "flip_data": None,
                "flip_targets": None,
                "original_fact": sampled_data
            }
        

if __name__ == "__main__":
    # test single task dataset
    dataset = SingleDataManager(data_dir="./dataset/single_fact_data")
    experiment_data = dataset.build_experiment_data(specific_samples=[0,1,2])
    print(experiment_data)

    # test multi task dataset
    sampled_data = {
        'chaining_tasks':[0,1], 
        'counting_tasks':[0,1], 
        'intersection_tasks': [0,1,2]
    }
    dataset = MultiDataManager(data_dir="./dataset/multi_fact_data")
    experiment_data = dataset.build_experiment_data(specific_samples=sampled_data)

    print(experiment_data)