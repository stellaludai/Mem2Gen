# TODO: clean up this file
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from stark_qa.skb import PrimeSKB

@dataclass
class Fact:
    """单个事实"""
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    
    def to_dict(self):
        return asdict(self)
    
    @property
    def meta_path(self):
        return (self.head_type, self.relation, self.tail_type)



@dataclass
class TaskCase:
    """任务实例"""
    task_id: str
    task_type: str
    memorization_tasks: List[Dict]
    generalization_tasks: Dict | List[Dict]
    facts: List[Fact]
    metadata: Dict = None
    
    def to_dict(self):
        """Convert TaskCase to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert Fact objects to dictionaries
        result['facts'] = [fact.to_dict() if hasattr(fact, 'to_dict') else asdict(fact) for fact in self.facts]
        return result


class BaseDatasetGenerator:
    """基础数据集生成器 - 提供共同的方法"""
    
    def __init__(self, kg: PrimeSKB):
        self.kg = kg
    
    def get_triples_by_relation(self, relation: str) -> List[Tuple]:
        """获取指定关系的三元组"""
        triples = []
        try:
            edge_ids = self.kg.get_edge_ids_by_type(relation)
            # 随机采样以提高效率
            if len(edge_ids) > 1000:
                edge_ids = random.sample(edge_ids, 1000)
            
            for edge_idx in edge_ids:
                try:
                    edge_info = self.kg.edge_index[:, edge_idx]
                    head_id = edge_info[0].item()
                    tail_id = edge_info[1].item()
                    
                    head_name = self.kg[head_id].name if hasattr(self.kg[head_id], 'name') else str(head_id)
                    tail_name = self.kg[tail_id].name if hasattr(self.kg[tail_id], 'name') else str(tail_id)
                    head_type = self.kg.get_node_type_by_id(head_id)
                    tail_type = self.kg.get_node_type_by_id(tail_id)
                    
                    triples.append((head_id, head_name, head_type, tail_id, tail_name, tail_type))
                except:
                    continue
        except:
            pass
        
        return triples
    