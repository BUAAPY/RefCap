from tqdm import tqdm
import os 
from abc import ABC
import json 

DENOISER_REGISTRY = {}

def REGISTER_DENOISER(names):
    def register_denoiser_cls(cls):
        if isinstance(names, str):
            if names in DENOISER_REGISTRY:
                raise ValueError(f"Cannot register duplicate denoiser ({names})")
            DENOISER_REGISTRY[names] = cls 
        elif isinstance(names, list):
            for name in names:
                if name in DENOISER_REGISTRY:
                    raise ValueError(f"Cannot register duplicate denoiser ({name})")
                DENOISER_REGISTRY[name] = cls
        return cls 
    return register_denoiser_cls

@REGISTER_DENOISER(["base"])
class BaseDenoiser(ABC):
    def __init__(self, cfg, models) -> None:
        self.cfg = cfg 
        self.it_sim_model = models["blip_itrtv_model"]
        self.it_sim_processor = models["blip_itrtv_processor"]

    
    def __call__(self, vid_list, captions, raw_scores):
        vid_2_cap = {x['vid_name']: x for x in captions}
        denoised_captions_save_file = os.path.join(self.cfg.exp_dir, self.cfg.denoised_captions_file)

        self.all_denoised_captions = []
        for vid in tqdm(vid_list, total=len(vid_list)):
            video_name = vid.split('.')[0]
            raw_cap = vid_2_cap[video_name]
            denoised_cap = self.denoise_caption(meta=raw_cap, raw_scores=raw_scores[video_name])
            self.all_denoised_captions.append(denoised_cap)
        
        with open(denoised_captions_save_file, "w+") as f:
            for cap in self.all_denoised_captions:
                cap_s = json.dumps(cap)
                f.writelines(f"{cap_s}\n")
        return self.all_denoised_captions
        

    def denoise_caption(self, meta, raw_scores):
        return meta # do nothing
    
def get_denoiser_class(name):
    if name not in DENOISER_REGISTRY:
        raise ValueError(f"Process name {name} not registered.")
    return DENOISER_REGISTRY[name]