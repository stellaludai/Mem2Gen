import json
import os
import random
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from openai import OpenAI
from stark_qa.skb import PrimeSKB
import tqdm
from dataset_generator_utils import Fact, TaskCase
from llm_single_fact_generator import AITaskGenerator, BaseDatasetGenerator
from utils.assistant_templates import (
    MEMORIZATION_TASK_SYSTEM_TEMPLATE,
    COUNTING_TASK_SYSTEM_TEMPLATE,
    CHAINING_TASK_SYSTEM_TEMPLATE,
    INTERSECTION_TASK_SYSTEM_TEMPLATE,
    MULTI_TASKS_USER_TEMPLATE
)


def number_to_english(value: int) -> str:
    """Convert small positive integers into simple English words for answers."""
    if value < 0:
        raise ValueError("Only non-negative integers are supported")

    ones = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
    }

    if value < 20:
        return ones[value]
    else:   
        return str(value)

class MultiTaskGenerator(AITaskGenerator):
    def __init__(self, base_url: Optional[str] = "https://api.deepseek.com", model: Optional[str] = "deepseek-chat"):
        super().__init__(base_url=base_url, model=model)

        self.system_message_dict = {
            "memorization": MEMORIZATION_TASK_SYSTEM_TEMPLATE,
            "counting": COUNTING_TASK_SYSTEM_TEMPLATE,
            "chaining": CHAINING_TASK_SYSTEM_TEMPLATE,
            "intersection": INTERSECTION_TASK_SYSTEM_TEMPLATE,
        }

        # default task type is memo
        self.system_message = self.system_message_dict["memorization"]

    def _triplets_to_message(self, triplets: List[Fact]) -> str:
        message = ''
        for i, triplet in enumerate(triplets):
            triplet_str = MULTI_TASKS_USER_TEMPLATE.format(
                i=str(i + 1),
                head=triplet.head,
                head_type=triplet.head_type,
                relation=triplet.relation,
                tail=triplet.tail,
                tail_type=triplet.tail_type,
            )
            message += triplet_str
        return message

    def set_task_type(self, task_type: str):
        if task_type in self.system_message_dict:
            self.system_message = self.system_message_dict[task_type]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def generate_questions(self, triplet_lists: List[List[Fact]]) -> List[Dict[str, Any]]:
        messages = [self._triplets_to_message(triplets) for triplets in triplet_lists]
        return self.generate_response_batch(messages)

class CountingDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, kg_root: str = None,):
        super().__init__(kg=PrimeSKB(root=kg_root))
        self.ai_generator = MultiTaskGenerator()

    def _sample_relation_head_groups(self, relation: str, num_heads: int = 5) -> List[List[Fact]]:
        
        all_triples = self.get_triples_by_relation(relation)
        
        head_to_tails = {}
        for head_id, head_name, head_type, tail_id, tail_name, tail_type in all_triples:
            if head_name not in head_to_tails:
                head_to_tails[head_name] = {
                    'head_type': head_type,
                    'tails': [],
                    'tail_type': tail_type
                }
            if tail_name not in head_to_tails[head_name]['tails']:
                head_to_tails[head_name]['tails'].append(tail_name)
        
        valid_heads = [(head, data) for head, data in head_to_tails.items() 
                      if len(data['tails']) >= 2]
        if len(valid_heads) < num_heads:
            print(f">>> Only {len(valid_heads)} valid heads for relation {relation}")
            
        # selected_heads = random.sample(valid_heads, min(num_heads, len(valid_heads)))
        random.shuffle(valid_heads)
        selected_head_num = min(num_heads, len(valid_heads))
        sampled_data = []
        for head, data in valid_heads:
            num_tails = random.randint(2, 4)
            if len(data['tails']) < num_tails:
                continue
            else:
                selected_tails = data['tails'][:num_tails]
                head_samples = []
                for tail in selected_tails:
                    head_samples.append(Fact(
                        head=head,
                        head_type=data['head_type'],
                        relation=relation,
                        tail=tail,
                        tail_type=data['tail_type']
                    ))
                sampled_data.append(head_samples)
            
            if len(sampled_data) >= selected_head_num:
                break
        return sampled_data
    
    def _sample_facts(self, total_sample: int) -> List[Fact]:
        available_relations = self.kg.RELATION_TYPES
        sampled_facts = []
        heads_per_relation = total_sample // len(available_relations) + 1

        for relation in available_relations:
            sampled_heads = self._sample_relation_head_groups(relation, num_heads=heads_per_relation)
            sampled_facts.extend(sampled_heads)

        print(f">>> Sampled a total of {len(sampled_facts)} facts.")
        return sampled_facts
    
    def generate_dataset(self, total_sample: Optional[int] = 100, output_dir: str = None) -> List[TaskCase]:
        sampled_facts = self._sample_facts(total_sample=total_sample)

        # memo tasks
        self.ai_generator.set_task_type("memorization")
        memo_questions_set = self.ai_generator.c(sampled_facts)
        
        # counting tasks
        self.ai_generator.set_task_type("counting")
        counting_facts = [[fact[0]] for fact in sampled_facts]
        counting_questions_set = self.ai_generator.generate_questions(counting_facts)
        print(counting_questions_set)
        
        # build dataset
        dataset = []
        assert len(memo_questions_set) == len(counting_questions_set)
        for facts, memo_questions, count_questions in zip(sampled_facts, memo_questions_set, counting_questions_set):
            memorization_tasks = []
            for fact, memo_question in zip(facts, memo_questions['questions']):
                memorization_tasks.append({
                    "prompt": memo_question,
                    "answer": fact.tail,
                    "task_type": "memorization"
                })

            count_value = len(facts)
            counting_tasks = [{
                "prompt": count_questions['question'],
                "answer": [str(count_value), number_to_english(count_value)],
                "task_type": "counting"
            }]

            task_case = TaskCase(
                task_id=str(uuid.uuid4()),
                task_type="counting",
                memorization_tasks=memorization_tasks,
                generalization_tasks=counting_tasks,
                facts=facts,
                metadata=None
            )
            dataset.append(task_case)
        
        # save
        self._save_dataset({"counting":dataset}, output_dir)


class ChainingDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, kg_root: str = None,):
        super().__init__(kg=PrimeSKB(root=kg_root))
        self.ai_generator = MultiTaskGenerator()

    def _sample_chain_facts(self, chain_length: int = 3, total_sample: int = 5) -> List[List[Fact]]:
        all_relations = list(self.kg.RELATION_TYPES)
        triples_by_relation = {
            relation: self.get_triples_by_relation(relation)
            for relation in all_relations
        }
        sampled_chains = []

        attempts = 0
        max_attempts = max(total_sample * 10, 1000)

        while len(sampled_chains) < total_sample:
            attempts += 1
            chain_facts: List[Fact] = []
            current_head_id: Optional[str] = None
            visited_entity_ids = set()  # ensures we never revisit entities within the same chain

            for _ in range(chain_length):
                triple = None
                relation_used = None

                for relation in random.sample(all_relations, len(all_relations)):
                    triples = triples_by_relation[relation]

                    if current_head_id is None:
                        valid_triples = [
                            t for t in triples
                            if t[3] != t[0] and t[3] not in visited_entity_ids
                        ]
                    else:
                        valid_triples = [
                            t for t in triples
                            if t[0] == current_head_id and t[3] != t[0] and t[3] not in visited_entity_ids
                        ]

                    if valid_triples:
                        triple = random.choice(valid_triples)
                        relation_used = relation
                        break

                if triple is None:
                    chain_facts = []
                    break

                head_id, head_name, head_type, tail_id, tail_name, tail_type = triple
                chain_facts.append(Fact(
                    head=head_name,
                    head_type=head_type,
                    relation=relation_used,
                    tail=tail_name,
                    tail_type=tail_type
                ))

                visited_entity_ids.add(head_id)
                visited_entity_ids.add(tail_id)
                current_head_id = tail_id

            if len(chain_facts) == chain_length:
                sampled_chains.append(chain_facts)

            if attempts >= max_attempts and len(sampled_chains) < total_sample:
                raise RuntimeError(
                    f"Unable to sample {total_sample} unique chains without loops after {attempts} attempts. "
                    "Consider reducing chain_length or total_sample."
                )
        
        print(f">>> Sampled a total of {len(sampled_chains)} chains.")
        return sampled_chains

    def generate_dataset(self, total_sample: Optional[int] = 100, output_dir: str = None) -> List[TaskCase]:
        sampled_facts = []
        sampled_facts.extend(self._sample_chain_facts(chain_length=2, total_sample= int(total_sample / 2)))
        sampled_facts.extend(self._sample_chain_facts(chain_length=3, total_sample= int(total_sample / 2)))

        # memo tasks
        self.ai_generator.set_task_type("memorization")
        memo_questions_set = self.ai_generator.generate_questions(sampled_facts)
        
        # chaining tasks
        self.ai_generator.set_task_type("chaining")
        chaining_questions_set = self.ai_generator.generate_questions(sampled_facts)
        
        # build dataset
        dataset = []
        assert len(memo_questions_set) == len(chaining_questions_set)
        for facts, memo_questions, chain_questions in zip(sampled_facts, memo_questions_set, chaining_questions_set):
            memorization_tasks = []
            for fact, memo_question in zip(facts, memo_questions['questions']):
                memorization_tasks.append({
                    "prompt": memo_question,
                    "answer": fact.tail,
                    "task_type": "memorization"
                })

            chaining_tasks = [{
                "prompt": chain_questions['question'],
                "answer": facts[-1].tail,
                "task_type": "chaining"
            }]

            task_case = TaskCase(
                task_id=str(uuid.uuid4()),
                task_type="chaining",
                memorization_tasks=memorization_tasks,
                generalization_tasks=chaining_tasks,
                facts=facts,
                metadata=None
            )
            dataset.append(task_case)

        # save
        self._save_dataset({"chaining":dataset}, output_dir)

    
class IntersectionDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, kg_root: str = None,):
        super().__init__(kg=PrimeSKB(root=kg_root))
        self.ai_generator = MultiTaskGenerator()

    def _sample_relation_tail_groups(self, relation: str, num_tails: int = 5) -> List[List[Fact]]:
        all_triples = self.get_triples_by_relation(relation)

        tail_to_heads: Dict[str, Dict[str, Any]] = {}
        for head_id, head_name, head_type, tail_id, tail_name, tail_type in all_triples:
            entry = tail_to_heads.setdefault(
                tail_name,
                {
                    "tail_type": tail_type,
                    "facts": [],
                    "seen_heads": set(),
                },
            )

            if head_name in entry["seen_heads"]:
                continue

            entry["facts"].append(
                Fact(
                    head=head_name,
                    head_type=head_type,
                    relation=relation,
                    tail=tail_name,
                    tail_type=tail_type,
                )
            )
            entry["seen_heads"].add(head_name)

        valid_tails = [
            (tail, data["facts"])
            for tail, data in tail_to_heads.items()
            if len(data["facts"]) >= 2
        ]

        if len(valid_tails) < num_tails:
            print(f">>> Only {len(valid_tails)} valid tails for relation {relation}")

        selected_tails_num = min(num_tails, len(valid_tails))
        random.shuffle(valid_tails)
        sampled_data: List[List[Fact]] = []
        for tail, facts in valid_tails:
            num_heads = random.randint(2, 4)
            if len(facts) < num_heads:
                continue
            else:
                sampled_data.append(facts[:num_heads])

            if len(sampled_data) >= selected_tails_num:
                break

        return sampled_data

    def _sample_facts(self, total_sample: int) -> List[List[Fact]]:
        available_relations = self.kg.RELATION_TYPES
        sampled_facts: List[List[Fact]] = []
        tails_per_relation = total_sample // len(available_relations) + 1

        for relation in available_relations:
            sampled_tails = self._sample_relation_tail_groups(
                relation, num_tails=tails_per_relation
            )
            sampled_facts.extend(sampled_tails)

        print(f">>> Sampled a total of {len(sampled_facts)} intersection fact groups.")
        return sampled_facts

    def generate_dataset(self, total_sample: Optional[int] = 100, output_dir: str = None) -> List[TaskCase]:
        sampled_facts = self._sample_facts(total_sample=total_sample)

        # memo tasks
        self.ai_generator.set_task_type("memorization")
        memo_questions_set = self.ai_generator.generate_questions(sampled_facts)

        # intersection tasks
        self.ai_generator.set_task_type("intersection")
        intersection_questions_set = self.ai_generator.generate_questions(sampled_facts)

        dataset: List[TaskCase] = []
        assert len(memo_questions_set) == len(intersection_questions_set)

        for facts, memo_questions, intersection_question in zip(
            sampled_facts, memo_questions_set, intersection_questions_set
        ):

            question_texts = memo_questions.get("questions")
            memorization_tasks = []
            for fact, memo_question in zip(facts, question_texts):
                memorization_tasks.append(
                    {
                        "prompt": memo_question,
                        "answer": fact.tail,
                        "task_type": "memorization",
                    }
                )

            shared_tail = facts[0].tail
            intersection_prompt = intersection_question.get("question")

            intersection_tasks = [
                {
                    "prompt": intersection_prompt,
                    "answer": shared_tail,
                    "task_type": "intersection",
                }
            ]

            task_case = TaskCase(
                task_id=str(uuid.uuid4()),
                task_type="intersection",
                memorization_tasks=memorization_tasks,
                generalization_tasks=intersection_tasks,
                facts=facts,
                metadata={
                    "shared_tail": shared_tail,
                    "num_heads": len(facts),
                },
            )
            dataset.append(task_case)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self._save_dataset({"intersection": dataset}, output_dir)

        return dataset
    



if __name__ == "__main__":
    # chaining
    generator = ChainingDatasetGenerator()
    generator.generate_dataset(total_sample=1000, output_dir="xxx")

    # intersection
    generator = IntersectionDatasetGenerator()
    generator.generate_dataset(total_sample=1000, output_dir="xxx")