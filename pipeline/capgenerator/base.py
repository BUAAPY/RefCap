import os 
from tqdm import tqdm 
from dataset.viddataset import VideoDatasetPerSec
from tqdm import tqdm 
from PIL import Image
from utils.basic_utils import load_jsonl

CAPGEN_REGISTRY = {}

def REGISTER_CAPGEN(names):
    def register_capgen_cls(cls):
        if isinstance(names, str):
            if names in CAPGEN_REGISTRY:
                raise ValueError(f"Cannot register duplicate capgener ({names})")
            CAPGEN_REGISTRY[names] = cls
        elif isinstance(names, list):
            for name in names:
                if name in CAPGEN_REGISTRY:
                    raise ValueError(f"Cannot register duplicate capgener ({name})")
                CAPGEN_REGISTRY[name] = cls 
        return cls 
    return register_capgen_cls


@REGISTER_CAPGEN(["base", "minigpt"])
class BaseCapGen:
    def __init__(self, cfg, models) -> None:
        self.cfg = cfg 
        self.models = models

        self.captions_save_file = os.path.join(self.cfg.meta_dir, self.cfg.captions_dir, f"{self.cfg.collection}_{self.cfg.caption_generator}.jsonl")
        if os.path.exists(self.captions_save_file):
            self.captions = load_jsonl(self.captions_save_file)
            self.already_video_names = set([x['vid_name'] for x in self.captions])
        else:
            self.captions = []
            self.already_video_names = set() 

    def __call__(self, vid_list):
        for vid in tqdm(vid_list,total=len(vid_list)):
            video_name = vid.split('.')[0]
            video_path = os.path.join(self.cfg.video_root, vid)
            print(f"\r{vid}", end="")
            if not os.path.isfile(video_path):
                print("video not exists!")
                continue
            self.generate_caption(video_name=video_name, video_path=video_path)
        return self.captions 
    
    def generate_caption(self, video_name, video_path):
        assert video_name in self.already_video_names, "You should use external scripts to generate captions first!"


def get_capgen_class(name):
    if name not in CAPGEN_REGISTRY:
        raise ValueError(f"Capgen name {name} not registered")
    return CAPGEN_REGISTRY[name]