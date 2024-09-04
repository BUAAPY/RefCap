from .base import REGISTER_DENOISER, BaseDenoiser
import os 
import copy 
import utils.basic_utils as basic_utils
from dataset.viddataset import VideoDatasetPerSec
import torch 
from PIL import Image 
from torch.nn.functional import normalize
import utils.sim_utils as sim_utils


@REGISTER_DENOISER(["window"])
class WindowSimDenoiser(BaseDenoiser):
    def __init__(self, cfg, models) -> None:
        super().__init__(cfg, models)
        self.it_sim_model = models["blip_itrtv_model"]
        self.it_sim_processor = models["blip_itrtv_processor"]
    

    def denoise_caption(self, meta, raw_scores):
        raw_scores = raw_scores.cpu().tolist()
        scores_dict = {} 
        for i, (fid, m) in enumerate(meta["frame_captions"].items()):
            sc = raw_scores[i]
            scores_dict[fid] = sc

        last_high_id = '0'
        for fid, m in meta['frame_captions'].items():
            sc = scores_dict[fid]
            if sc > self.cfg.figsim_denoise_thr:
                last_high_id = fid 
            else:
                if last_high_id != '0' and (int(fid)-int(last_high_id))<=self.cfg.denoise_window_width:
                    meta['frame_captions'][fid] = copy.deepcopy(meta['frame_captions'][last_high_id])
        return meta 
    