from tqdm import tqdm
import os 
import utils.basic_utils as basic_utils
from dataset.dataset import DataSet4Test
from pipeline.treebuilder.capTree import CapTree

from abc import ABC, abstractmethod

RETRIEVERPIPE_REGISTRY = {}

def REGISTER_RETRIEVEPIPE(names):
    def register_retrieverpipe_cls(cls):
        if isinstance(names, str):
            if names in RETRIEVERPIPE_REGISTRY:
                raise ValueError(f"Cannot register duplicate retrievepipe ({names})")
            RETRIEVERPIPE_REGISTRY[names] = cls 
        elif isinstance(names, list):
            for name in names:
                if name in RETRIEVERPIPE_REGISTRY:
                    raise ValueError(f"Cannot register duplicate retrievepipe ({name})")
                RETRIEVERPIPE_REGISTRY[name] = cls
        return cls 
    return register_retrieverpipe_cls

@REGISTER_RETRIEVEPIPE(["base"])
class BaseRetrievePipe(ABC):
    def __init__(self, cfg, captree: CapTree, models) -> None:
        self.cfg = cfg 
        self.captree = captree
    
    @abstractmethod
    def retrieval(self, test_dataset: DataSet4Test):
        pass
    
def get_retrievepipe_class(name):
    if name not in RETRIEVERPIPE_REGISTRY:
        raise ValueError(f"RetrievePpipe name {name} not registered.")
    return RETRIEVERPIPE_REGISTRY[name]