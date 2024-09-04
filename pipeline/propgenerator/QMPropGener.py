import torch 
import torch.nn.functional as F
import numpy as np 
import os 
import json 
from tqdm import tqdm 

from sentence_transformers import util as sim_util 

from .base import BasePropGen, REGISTER_PROPGEN

import utils.sim_utils as sim_utils
import utils.basic_utils as basic_utils
import utils.tree_utils as tree_utils
import spacy 


@REGISTER_PROPGEN(["qm"])
class QMPropGenerator(BasePropGen):
    def __init__(self, cfg, models) -> None:
        self.cfg = cfg 
        self.txt_sim_model = models['sentence_transformer']
        self.it_sim_model = models['blip_itrtv_model']
        self.it_sim_processor = models['blip_itrtv_processor']
        self.nlp = spacy.load('en_core_web_sm')
    

    def calculate_similarities(self, sentences, capframe_scores, frame_features):
        if self.cfg.prop_sim_type == "vis":
            # frame_features = torch.load(os.path.join(self.cfg.meta_dir, self.cfg.framefeatures_dir, self.cfg.collection, vid, self.cfg.framefeatures_file)).to(self.cfg.device)
            sims = sim_util.cos_sim(frame_features, frame_features)
        else:
            with torch.no_grad():
                embeddings = self.txt_sim_model.encode(sentences)
                txt_sims = sim_util.cos_sim(embeddings, embeddings).to(self.cfg.device)
                if self.cfg.prop_sim_type == 'txt': # only text
                    sims = txt_sims
                elif self.cfg.prop_sim_type == 'it': # with image-text similarities
                    sims = txt_sims * capframe_scores[None, :] * capframe_scores[:, None]
                else:
                    raise NotImplementedError()
        return sims 
    
    def __call__(self, vid_list, captions, scores, all_frame_features):
        proposals = {}  
        vid_2_cap = {x['vid_name']: x for x in captions}
        prop_sims = {} 
        for vid in tqdm(vid_list, total=len(vid_list)):
            video_name = vid.split('.')[0]
            cap = vid_2_cap[video_name]
            sentences = [m['cap'] for v,m in cap['frame_captions'].items()]
            capframe_scores = scores[video_name].to(self.cfg.device)
            frame_features = all_frame_features[video_name].to(self.cfg.device)
             
            sims = self.calculate_similarities(sentences, capframe_scores, frame_features)
            prop_sims[video_name] = sims.cpu()
            sims = sims[None, None, :, :].to(self.cfg.device) # [1,1,H,H]

            proposals[video_name] = {
                'proposals': self.generate_proposal(sims, cap, capframe_scores),
                'duration': cap['duration']
            }
        torch.save(prop_sims, os.path.join(self.cfg.exp_dir, self.cfg.prop_sim_path))
        basic_utils.save_json(proposals, os.path.join(self.cfg.exp_dir, self.cfg.proposals_file))
        
        return proposals 


    def generate_proposal(self, sims, metas, capframe_scores):
        VID_LEN = sims.shape[-1]
        duration = metas['duration']
        assert capframe_scores.shape[0] == VID_LEN

        kernel_size = 2*self.cfg.prop_kernel_width+1
        weight = torch.zeros(kernel_size, kernel_size).float().to(self.cfg.device)
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i==self.cfg.prop_kernel_width or j==self.cfg.prop_kernel_width:
                    continue
                if (i<self.cfg.prop_kernel_width and j<self.cfg.prop_kernel_width) or (i>self.cfg.prop_kernel_width and j>self.cfg.prop_kernel_width):
                    weight[i][j] = 1. 
                else:
                    weight[i][j] = -1. 
        weight = weight[None, None, :, :] # [1,1,k,k]
        boundaries = [] 
        scores = F.conv2d(sims, weight=weight, stride=[1,1], padding=self.cfg.prop_kernel_width)
        scores = torch.diagonal(scores.reshape(scores.shape[-2], scores.shape[-1]))
        scores = (scores-scores.min())/(scores.max()-scores.min())
        scores, indices = torch.sort(scores, descending=True)
        scores = scores.cpu().tolist()
        indices = indices.cpu().tolist()
        for sc, id in zip(scores, indices):
            if id in [0, VID_LEN-1]:
                continue
            if len(boundaries)==0:
                boundaries.append(id)
                continue
            if len(boundaries)+1>=self.cfg.prop_max_cnt:
                break 
            # check redundancy
            tmp = [abs(t-id) for t in boundaries]
            if min(tmp) < self.cfg.prop_kernel_width:
                continue
        
            if sc<self.cfg.prop_score_thr and len(boundaries)+1>=self.cfg.prop_min_cnt:
                break 
            boundaries.append(id)
        boundaries.sort()
        if len(boundaries) == 0: # use the whole video
            boundaries = [0, VID_LEN]
        else:
            if boundaries[0] < self.cfg.min_prop_size:
                boundaries[0] = 0 
            else:
                boundaries = [0] + boundaries
            
            if (VID_LEN-boundaries[-1]) < self.cfg.min_prop_size:
                boundaries[-1] = VID_LEN
            else:
                boundaries = boundaries + [VID_LEN]
        # boundaries = [0] + boundaries + [VID_LEN] # [,)
        n = len(boundaries)

        res = [] 
        for i in range(n-1):
            st_idx, ed_idx = boundaries[i], boundaries[i+1]
            st, ed = (st_idx/VID_LEN)*duration, (ed_idx/VID_LEN)*duration 
            scores = capframe_scores[st_idx:ed_idx]
            # max_val = torch.max(scores)
            max_idx = torch.argmax(scores).item()
            max_idx += st_idx 
            # can add all keys here!
            cap = metas['frame_captions'][str(max_idx)]['cap']
            keys = []
            for idx in range(st_idx, ed_idx):
                nouns, verbs = tree_utils.get_nouns_verbs(self.nlp, metas['frame_captions'][str(idx)]['cap'])
                keys += nouns 
                keys += verbs 
            keys = list(set(keys))
            res.append({'st': st, 'ed': ed, 'cap': cap, 'keys': keys})
        return res