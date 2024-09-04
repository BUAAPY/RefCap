from tqdm import tqdm
import os 
import utils.basic_utils as basic_utils
from abc import ABC, abstractmethod

PROPGEN_REGISTRY = {}

def REGISTER_PROPGEN(names):
    def register_propgen_cls(cls):
        if isinstance(names, str):
            if names in PROPGEN_REGISTRY:
                raise ValueError(f"Cannot register duplicate proposal generator ({names})")
            PROPGEN_REGISTRY[names] = cls 
        elif isinstance(names, list):
            for name in names:
                if name in PROPGEN_REGISTRY:
                    raise ValueError(f"Cannot register duplicate proposal generator ({name})")
                PROPGEN_REGISTRY[name] = cls
        return cls 
    return register_propgen_cls

@REGISTER_PROPGEN(["base"])
class BasePropGen(ABC):
    def __init__(self, cfg, models) -> None:
        self.cfg = cfg 
    
    @abstractmethod
    def __call__(self, vid_list, captions, scores):
        pass 
    
def get_propgen_class(name):
    if name not in PROPGEN_REGISTRY:
        raise ValueError(f"Propgenerator name {name} not registered.")
    return PROPGEN_REGISTRY[name]