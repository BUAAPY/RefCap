from .base import REGISTER_RETRIEVEPIPE, BaseRetrievePipe
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from pipeline.treebuilder.capTree import CapTree

from typing import * 
from tqdm import tqdm

from sentence_transformers import util as sent_util

import utils.tree_utils as tree_utils
import functools

from dataset.dataset import DataSet4Test

import json 
import spacy 
from torchtext.vocab import Vectors

import utils.sim_utils as sim_utils
import time 

@REGISTER_RETRIEVEPIPE(["mix"])
class MixInferPipe(BaseRetrievePipe):
    def __init__(self, cfg, captree: CapTree, models) -> None:
        super().__init__(cfg, captree, models)
        self.captree = captree
        self.cfg = cfg 
        self.nlp = spacy.load("en_core_web_sm")
        self.vecs = Vectors(cfg.glove_model, cache=cfg.meta_dir)
    
    def compute_similarities(self, word_list_1, word_list_2):
        def get_vecs(word_list):
            tmp_list = []
            for word in word_list:
                if word not in self.vecs.stoi:
                    continue
                tmp_list.append(self.vecs.stoi[word])
            if len(tmp_list) == 0:
                return None 
            return self.vecs.vectors[tmp_list].numpy()
        
        vec1 = get_vecs(word_list_1)
        vec2 = get_vecs(word_list_2)
        if vec1 is None or vec2 is None:
            return 0. 
        dot = np.dot(vec1, vec2.T)  # [N1, N2]
        sims = dot / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(
            np.sum(vec2 ** 2))
        sims = torch.tensor(sims).to(self.cfg.device)
        return sims 
    
    def retrieval(self, test_dataset: DataSet4Test):
        
        dataloader = DataLoader(test_dataset, batch_size=self.cfg.eval_query_bsz, num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True)
        vr_scores = [] 
        vcmr_res = [] 
        for idx, batch in tqdm(enumerate(dataloader), desc="Evaluating...", total=len(dataloader)):
            # print(f"desc_name: {batch[0][0]}, desc: {batch[1][0]}, vid_name: {batch[2][0]}")
            desc_names, desc_ids, gt_vid_names, gt_vid_ids, descs, starts, ends = batch 
            # import pdb; pdb.set_trace() 
            Nq, Nv, Np = len(descs), len(self.captree.vidname_to_capids), self.captree.cap_cnt
            """
                Compute sentence-similarities
            """
            sent_scores = torch.zeros((Nq, Nv)).to(self.cfg.device)
            desc_features = self.captree.encode_caps(descs) # cap features
            sent_sims = sent_util.cos_sim(desc_features, self.captree.cap_features) # [Nq, Np]
            for vid_name, cap_ids in self.captree.vidname_to_capids.items(): # cap_ids: List[int]
                vid_id = test_dataset.vid_name_to_id[vid_name] 
                rel_sims = sent_sims[:, cap_ids] # [Nq, Nr]
                sims = torch.max(rel_sims, dim=1)[0] # [Nq]
                sent_scores[:, vid_id] = sims 

            """
                Compute keyword-similarities
            """
            descs_keys = list(map(lambda x: tree_utils.extract_keys(self.nlp, self.captree.key_feature_model.stoi, x, 10), descs)) # List[List[str]]
            all_descs_keys = functools.reduce(lambda x,y: x+y, descs_keys)
            descs_keys_cnts = list(map(lambda x: len(x), descs_keys))
            key_sims = self.compute_similarities(word_list_1=all_descs_keys, word_list_2=self.captree.keys) # [Nkq, Nkt]

            """
                VCMR first, PRVR second (max(Video)->max(max(Event)))
            """
            
            max_QN = max(descs_keys_cnts)
            max_PN = max(self.captree.prop_key_cnts)
            S = torch.zeros((Nq*max_QN, Np*max_PN)).to(self.cfg.device)
            # import pdb; pdb.set_trace() 

            I_Q = functools.reduce(lambda x,y: x+y, [list(range(i*max_QN, i*max_QN+descs_keys_cnts[i])) for i in range(len(descs_keys_cnts))])
            I_N = functools.reduce(lambda x,y: x+y, [list(range(i*max_PN, i*max_PN+self.captree.prop_key_cnts[i])) for i in range(len(self.captree.prop_key_cnts))])
            I_Q, I_N = torch.tensor(I_Q), torch.tensor(I_N)
            X, Y = torch.cartesian_prod(I_Q, I_N).split(1, dim=1)
            X, Y = X.squeeze().to(self.cfg.device), Y.squeeze().to(self.cfg.device)

            S[X, Y] = key_sims.reshape(-1)

            S_perquery_perevent = S.reshape(Nq, max_QN, Np, max_PN,).permute(0,2,1,3) # [Nq, Np, max_QN, max_PN]
            S_perquery_maxevent = S_perquery_perevent.max(dim=3)[0] # [Nq, Np, max_QN]
            if self.cfg.key_policy == "max_mean":
                key_sims_per_proposal = S_perquery_maxevent.sum(dim=2) / torch.tensor(descs_keys_cnts).to(self.cfg.device).unsqueeze(-1)
            elif self.cfg.key_policy == "max_max":
                key_sims_per_proposal = S_perquery_maxevent.max(dim=2)[0]
            else:
                raise NotImplementedError
            
            S_maxevent = S_perquery_maxevent.permute(0,2,1).reshape(Nq*max_QN, Np)
            max_prop_per_videos = max(self.captree.prop_cnt_per_video)
            V = torch.zeros((Nq*max_QN, Nv*max_prop_per_videos)).to(self.cfg.device)
            I_P = functools.reduce(lambda x,y: x+y, [list(range(i*max_prop_per_videos, i*max_prop_per_videos+self.captree.prop_cnt_per_video[i])) for i in range(len(self.captree.prop_cnt_per_video))])
            I_P = torch.tensor(I_P)
            I_Q_ = torch.arange(V.shape[0])
            X, Y = torch.cartesian_prod(I_Q_, I_P).split(1, dim=1)
            X, Y = X.squeeze().to(self.cfg.device), Y.squeeze().to(self.cfg.device)
            V[X,Y] = S_maxevent.reshape(-1)
            V_perquery_pervideo = V.reshape(Nq, max_QN, Nv, max_prop_per_videos).permute(0,2,1,3) # [Nq, Nv, max_QN, max_prop_per_videos]
            V_perquery_maxvideo = V_perquery_pervideo.max(dim=3)[0]
            if self.cfg.key_policy == "max_mean":
                key_scores = V_perquery_maxvideo.sum(dim=2)/torch.tensor(descs_keys_cnts).unsqueeze(-1).to(self.cfg.device)
            elif self.cfg.key_policy == "max_max":
                key_scores = V_perquery_maxvideo.max(dim=2)[0]


            # VCMR
            sent_sims = sim_utils.normalize_min_max(sent_sims, dim=1)
            key_sims_per_proposal = sim_utils.normalize_min_max(key_sims_per_proposal, dim=1) # [Nq, Nv]
            esm_sims = sent_sims * self.cfg.retrieve_sent_ratio + key_sims_per_proposal * (1- self.cfg.retrieve_sent_ratio)

            # PRVR
            sent_scores = sim_utils.normalize_min_max(sent_scores, dim=1)
            key_scores = sim_utils.normalize_min_max(key_scores, dim=1) # [Nq, Np]
            scores = sent_scores * self.cfg.retrieve_sent_ratio + key_scores * (1-self.cfg.retrieve_sent_ratio)
            vr_scores.append(scores)

            max_vcmr_prop_cnts = min(self.cfg.max_vcmr_props, Np)
            topk_cap_sims, topk_cap_ids = torch.topk(esm_sims, max_vcmr_prop_cnts, dim=1) # [Nq, 1000]
            
            for i in range(Nq):
                sel_cap_ids = topk_cap_ids[i]# [1000]
                sel_cap_sims = topk_cap_sims[i].tolist() 
                sel_node_ids = self.captree.cap_to_nodeid[sel_cap_ids].tolist()
                preds = [] 
                for node_id, score in zip(sel_node_ids, sel_cap_sims):
                    node = self.captree.nodeid_to_node[node_id]
                    vid_name = node['vid_name']
                    vid_id = test_dataset.vid_name_to_id[vid_name]
                    st, ed = node['st'], node['ed']
                    cap = node['caps'][0]
                    preds.append([
                        vid_id, st, ed, score, vid_name, cap 
                    ])
                preds = dict(
                    desc_id = desc_ids[i].item(), 
                    desc_name = desc_names[i], 
                    desc = descs[i], 
                    gt_vid_name=gt_vid_names[i],
                    gt_ts=[starts[i].item(), ends[i].item()],
                    predictions = preds 
                )
                # import pdb; pdb.set_trace() 
                vcmr_res.append(preds)

        vr_scores = torch.cat(vr_scores, dim=0)  # [Nq, Nv]
        vr_indices = torch.argsort(vr_scores, dim=1, descending=True).cpu().numpy().copy()

        vcmr_res_dict = dict(
            VCMR=vcmr_res,
            video2idx = test_dataset.vid_name_to_id
        )


        return vr_indices, vcmr_res_dict