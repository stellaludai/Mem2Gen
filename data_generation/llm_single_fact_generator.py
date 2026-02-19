import json
import os
import random
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from stark_qa.skb import PrimeSKB
import tqdm
from data_generation.dataset_generator_utils import Fact, TaskCase
from utils.assistant_templates import (
    # SINGLE_TASK_SYSTEM_TEMPLATE_2,
    SINGLE_TASK_SYSTEM_TEMPLATE,
    SINGLE_TASK_USER_TEMPLATE
)

# TODO: the gold answer for multilingual should be translated?

class AITaskGenerator:
    def __init__(self, base_url: Optional[str] = "https://api.deepseek.com", model: Optional[str] = "deepseek-chat"):
        """
        Initialize the OpenAI question generator.

        """
        assert os.getenv("OPENAI_API_KEY") is not None, "Please set the OPENAI_API_KEY environment variable."
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = model
        self.system_message = None
        
    def generate_response(self, message) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": message}
                    ],
                    temperature=1.3,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                return json.loads(content)

            except Exception as e:
                last_error = e
                print(f"Error generating questions (attempt {attempt + 1}/5): {e}")
                time.sleep(min(2 ** attempt, 10))

        print(f"Error generating questions after 5 attempts: {last_error}")
        return {}

    def generate_response_batch(self, messages: List[str], max_workers: int = 10) -> List[Dict[str, Any]]:

        results = [None] * len(messages)  # Pre-allocate results list to maintain order
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate_response, message): i 
                for i, message in enumerate(messages)
            }
            
            # Process completed futures with progress bar
            with tqdm.tqdm(total=len(messages), desc="Generating responses") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        print(f"Error processing message {index}: {e}")
                        results[index] = {}
                    finally:
                        pbar.update(1)
        
        return results
    
class BaseDatasetGenerator:
    def __init__(self, kg: PrimeSKB):
        self.kg = kg
    
    def get_triples_by_relation(self, relation: str) -> List[Tuple]:
        triples = []
        edge_ids = self.kg.get_edge_ids_by_type(relation)
        if len(edge_ids) > 1000:
            edge_ids = random.sample(edge_ids, 1000)
        
        for edge_idx in edge_ids:
            edge_info = self.kg.edge_index[:, edge_idx]
            head_id = edge_info[0].item()
            tail_id = edge_info[1].item()
            
            head_name = self.kg[head_id].name if hasattr(self.kg[head_id], 'name') else str(head_id)
            tail_name = self.kg[tail_id].name if hasattr(self.kg[tail_id], 'name') else str(tail_id)
            head_type = self.kg.get_node_type_by_id(head_id)
            tail_type = self.kg.get_node_type_by_id(tail_id)
            
            triples.append((head_id, head_name, head_type, tail_id, tail_name, tail_type))
        
        return triples
    
    def _save_dataset(self, dataset: Dict, output_dir: str,):
        summary_counts: Dict[str, int] = {}
        unique_task_ids = set()

        # save individual files and collect summary statistics
        for task_type, tasks in dataset.items():
            task_file = os.path.join(output_dir, f"{task_type}_tasks.json")
            with open(task_file, "w", encoding="utf-8") as f:
                # Convert TaskCase objects to dictionaries for JSON serialization
                serializable_tasks = []
                for task in tasks:
                    task_id = getattr(task, "task_id", None)
                    if task_id is None and isinstance(task, dict):
                        task_id = task.get("task_id")
                    if task_id is not None:
                        unique_task_ids.add(task_id)

                    if hasattr(task, 'to_dict'):
                        serializable_tasks.append(task.to_dict())
                    else:
                        serializable_tasks.append(task)
                # save the entire task list
                json.dump(serializable_tasks, f, ensure_ascii=False, indent=2)


            summary_key = f"{task_type}_tasks"
            summary_counts[summary_key] = summary_counts.get(summary_key, 0) + len(tasks)

        summary_data = {
            "generation_time": datetime.now().isoformat(),
            "available_tasks": summary_counts,
            "total_tasks": len(unique_task_ids)
        }

        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

class SingleTaskGenerator(AITaskGenerator):
    def __init__(self, base_url: Optional[str] = "https://api.deepseek.com", model: Optional[str] = "deepseek-chat"):
        super().__init__(base_url=base_url, model=model)
        self.system_message = SINGLE_TASK_SYSTEM_TEMPLATE
        
    def _triplet_to_message(self, triplet: Fact) -> str:
        return SINGLE_TASK_USER_TEMPLATE.format(
            head=triplet.head,
            head_type=triplet.head_type,
            relation=triplet.relation,
            tail=triplet.tail,
            tail_type=triplet.tail_type
        )
    
    def generate_questions(self, triplets: List[Fact]) -> List[Dict[str, Any]]:
        messages = [self._triplet_to_message(triplet) for triplet in triplets]
        return self.generate_response_batch(messages)

class SingleFactDatasetGenerator(BaseDatasetGenerator):
    """单事实数据集生成器"""
    
    def __init__(self, kg_root: str = None,):
        super().__init__(kg=PrimeSKB(root=kg_root))
        self.ai_generator = SingleTaskGenerator()
    
    def _sample_fact(self, total_sample: Optional[int] = None, sample_per_relation: Optional[int] = None) -> List[Fact]:
        """
        从知识图谱中抽样事实
        Args:
            total_sample: 总抽样数量，若不为None则尽量平均从各关系中抽取
            sample_per_relation: 每个关系的抽样数量，若不为None则从各关系中分别抽取
        Returns:
            List[Fact]: 抽样得到的事实列表
        """
        # 检查参数有效性
        if total_sample is None and sample_per_relation is None:
            raise ValueError("total_sample and sample_per_relation cannot both be None")
        if total_sample is not None and sample_per_relation is not None:
            raise ValueError("total_sample and sample_per_relation cannot both be provided")
        
        # 获取所有可用关系
        available_relations = self.kg.RELATION_TYPES
        sampled_facts = []
        
        if sample_per_relation is not None:
            # 从各关系中分别抽取指定数量
            print(f"Sampling {sample_per_relation} facts per relation from {len(available_relations)} relations...")
            for relation in available_relations:
                triples = self.get_triples_by_relation(relation)
                if not triples:
                    print(f"No triples found for relation: {relation}")
                    continue
                
                # 随机抽样
                sampled_triples = random.sample(triples, min(sample_per_relation, len(triples)))
                
                # 转换为Fact对象
                for head_id, head_name, head_type, tail_id, tail_name, tail_type in sampled_triples:
                    fact = Fact(
                        head=head_name,
                        head_type=head_type,
                        relation=relation,
                        tail=tail_name,
                        tail_type=tail_type
                    )
                    sampled_facts.append(fact)
                
                print(f"Sampled {len(sampled_triples)} facts from relation: {relation}")
        
        elif total_sample is not None:
            print(f"Sampling {total_sample} facts evenly from {len(available_relations)} relations...")
            
            base_per_relation = total_sample // len(available_relations)
            remainder = total_sample % len(available_relations)
            
            relation_sample_counts = [base_per_relation] * len(available_relations)
            for i in range(remainder):
                relation_sample_counts[i] += 1
            
            for i, relation in enumerate(available_relations):
                target_count = relation_sample_counts[i]
                if target_count == 0:
                    continue
                
                triples = self.get_triples_by_relation(relation)
                if not triples:
                    print(f"No triples found for relation: {relation}")
                    continue
                
                actual_count = min(target_count, len(triples))
                sampled_triples = random.sample(triples, actual_count)
                
                for head_id, head_name, head_type, tail_id, tail_name, tail_type in sampled_triples:
                    fact = Fact(
                        head=head_name,
                        head_type=head_type,
                        relation=relation,
                        tail=tail_name,
                        tail_type=tail_type
                    )
                    sampled_facts.append(fact)
                
                print(f"Sampled {actual_count}/{target_count} facts from relation: {relation}")
        
        print(f"Total sampled facts: {len(sampled_facts)}")
        return sampled_facts

    def _questions_to_tasks(self, questions: List[Dict[str, Any]], triplet: Fact) -> Dict[str, Any]:
        task_id = f"{triplet.head}_{triplet.relation}_{triplet.tail}"
        memorize_tasks = [{
            "prompt": questions.get("memorize", ""),
            "answer": triplet.tail,
            "task_type": "memorization",
        }]

        # paraphase tasks
        temp_paraphrase_tasks = []
        for q in questions.get("paraphrase", []):
            temp_paraphrase_tasks.append({
                "prompt": q,
                "answer": triplet.tail,
                "task_type": "paraphrase",
            })
        paraphase_tasks = TaskCase(task_id=task_id, 
                            task_type="paraphrase",
                            memorization_tasks=memorize_tasks,
                            generalization_tasks=temp_paraphrase_tasks,
                            facts=[triplet])
        
        # reverse tasks
        temp_reverse_tasks = [{
            "prompt": questions.get("reverse", ""),
            "answer": triplet.head,
            "task_type": "reverse",
        }]
        reverse_tasks = TaskCase(task_id=task_id,
                                        task_type="reverse",
                                        memorization_tasks=memorize_tasks,
                                        generalization_tasks=temp_reverse_tasks,
                                        facts=[triplet])
        
        # fact check
        fact_check_template = "Decide whether the following statement is true or false, answer with 'true' or 'false' ONLY. Statement: {statement}"
        temp_fact_check_tasks = []
        true_statement = questions.get("fact_checking", {}).get("true", "")
        false_statement = questions.get("fact_checking", {}).get("false", "")
        temp_fact_check_tasks.append({
            "prompt": fact_check_template.format(statement=true_statement),
            "answer": "true",
            "task_type": "fact_checking_true",
        })
        temp_fact_check_tasks.append({
            "prompt": fact_check_template.format(statement=false_statement),
            "answer": "false",
            "task_type": "fact_checking_false",
        })
        fact_check_tasks = TaskCase(task_id=task_id,
                                        task_type="fact_checking",
                                        memorization_tasks=memorize_tasks,
                                        generalization_tasks=temp_fact_check_tasks,
                                        facts=[triplet])
        
        # reverse fact check
        temp_reverse_fact_check_tasks = []
        true_statement = questions.get("reverse_fact_checking", {}).get("true", "")
        false_statement = questions.get("reverse_fact_checking", {}).get("false", "")
        temp_reverse_fact_check_tasks.append({
            "prompt": fact_check_template.format(statement=true_statement),
            "answer": "true",
            "task_type": "reverse_fact_checking_true",
        })
        temp_reverse_fact_check_tasks.append({
            "prompt": fact_check_template.format(statement=false_statement),
            "answer": "false",
            "task_type": "reverse_fact_checking_false",
        })
        reverse_fact_check_tasks = TaskCase(task_id=task_id,
                                            task_type="reverse_fact_checking",
                                            memorization_tasks=memorize_tasks,
                                            generalization_tasks=temp_reverse_fact_check_tasks,
                                            facts=[triplet])
        
        # # cross-lingual tasks
        # temp_crosslingual_tasks = []
        # for lang, q in questions.get("crosslingual", {}).items():
        #     temp_crosslingual_tasks.append({
        #         "prompt": q,
        #         "answer": triplet.tail,
        #         "task_type": f"cross_lingual_{lang}",
        #     })
        # crosslingual_tasks = TaskCase(task_id=task_id,
        #                                     task_type="crosslingual",
        #                                     memorization_tasks=memorize_tasks,
        #                                     generalization_tasks=temp_crosslingual_tasks,
        #                                     facts=[triplet])

        # cross-lingual tasks
        crosslingual_tasks = {}
        for lang, q in questions.get("crosslingual", {}).items():
            crosslingual_tasks["crosslingual_"+lang] = TaskCase(
                task_id=task_id,
                task_type=f"crosslingual_{lang}",
                memorization_tasks=memorize_tasks,
                generalization_tasks=[{
                    "prompt": q,
                    "answer": triplet.tail,
                    "task_type": f"cross_lingual_{lang}",
                }],
                facts=[triplet]
            )


        
        return {'paraphrase': paraphase_tasks,
                'reverse': reverse_tasks,
                'fact_checking': fact_check_tasks,
                'reverse_fact_checking': reverse_fact_check_tasks,
                # 'crosslingual': crosslingual_tasks
                **crosslingual_tasks}



    def generate_dataset(self, 
                           total_sample: Optional[int] = 100,
                           sample_per_relation: Optional[int] = None,
                           output_dir: str = None) -> Dict:
        """Generate complete LLM-based single fact dataset
        
        Args:
            total_sample: Total number of triplets to sample and generate questions for
            sample_per_relation: Number of triplets to sample per relation
            output_dir: Output directory for saved files
            
        Returns:
            Generated dataset dictionary
        """
        print("Starting LLM-based single fact dataset generation...")
        
        if total_sample is not None:
            print(f"Parameters: total_sample={total_sample}")
        elif sample_per_relation is not None:
            print(f"Parameters: sample_per_relation={sample_per_relation}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample triplets from knowledge graph
        triplets = self._sample_fact(total_sample=total_sample, sample_per_relation=sample_per_relation)
        
        # Generate questions for each triplet
        generated_questions = []
        failed_count = 0
        
        print(f">>> Generating {len(triplets)} questions with LLM...")
        
        # Use batch processing for concurrent question generation
        questions_list = self.ai_generator.generate_questions(triplets)
        
        # Process results and pair with triplets
        for i, (triplet, questions) in enumerate(zip(triplets, questions_list)):
            if questions:  # If questions were successfully generated
                generated_questions.append((triplet, questions))
            else:
                failed_count += 1

        print(f">>> Generation complete. Failed count: {failed_count}")
        print(">>> parsing generated questions into task cases...")

        # parse generated questions into task cases
        dataset: Dict[str, List[TaskCase]] = {}
        for triplet, questions in generated_questions:
            for task_type, task_case in self._questions_to_tasks(questions, triplet).items():
                dataset.setdefault(task_type, []).append(task_case)

        # save dataset to output directory
        if output_dir is not None:
            self._save_dataset(dataset, output_dir)


if __name__ == "__main__":
    output_dir = "xxx"
    generator = SingleFactDatasetGenerator()
    dataset = generator.generate_dataset(total_sample=5000, output_dir=output_dir)
    print(">>> Dataset generation completed.")