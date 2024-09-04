from .base import REGISTER_RETRIEVEPIPE, BaseRetrievePipe
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from pipeline.treebuilder.capTree import CapTree

from typing import * 
from tqdm import tqdm

from sentence_transformers import util as sent_util
from dataset.dataset import DataSet4Test


@REGISTER_RETRIEVEPIPE(["sent"])
class SentInferPipe(BaseRetrievePipe):
    def retrieval(self, test_dataset: DataSet4Test):
        dataloader = DataLoader(test_dataset, batch_size=self.cfg.eval_query_bsz, num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True)
        vr_scores = [] 
        vcmr_res = [] 
        for idx, batch in tqdm(enumerate(dataloader), desc="Evaluating...", total=len(dataloader)):
            # print(f"desc_name: {batch[0][0]}, desc: {batch[1][0]}, vid_name: {batch[2][0]}")
            desc_names, desc_ids, gt_vid_names, gt_vid_ids, descs, starts, ends = batch 
            # import pdb; pdb.set_trace() 
            desc_features = self.captree.encode_caps(descs)
            # print(desc_features.shape)
            cos_sims = sent_util.cos_sim(desc_features, self.captree.cap_features) # [Nq, Nc]  和所有的caption的相似度(caption数量大于视频数量)
            Nq, Nv = len(descs), len(self.captree.vidname_to_capids)

            scores = torch.zeros((Nq, Nv)).to(self.cfg.device)
            for vid_name, cap_ids in self.captree.vidname_to_capids.items(): # cap_ids: List[int]
                vid_id = test_dataset.vid_name_to_id[vid_name] 
                rel_sims = cos_sims[:, cap_ids] # [Nq, Nr]
                """
                    Max-Strategy For Coarse Retrieval
                """
                sims = torch.max(rel_sims, dim=1)[0] # [Nq]
                scores[:, vid_id] = sims 
            vr_scores.append(scores)

            # Moment-Retrieval
            max_vcmr_prop_cnts = min(self.cfg.max_vcmr_props, cos_sims.shape[1])
            topk_cap_sims, topk_cap_ids = torch.topk(cos_sims, max_vcmr_prop_cnts, dim=1) # [Nq, 1000]
            
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
                    preds.append([
                        vid_id, st, ed, score, vid_name
                    ])
                preds = dict(
                    desc_id = desc_ids[i].item(), 
                    desc_name = desc_names[i], 
                    desc = descs[i], 
                    gt_vid_name=gt_vid_names[i],
                    gt_ts=[starts[i].item(), ends[i].item()],
                    predictions = preds 
                )
                vcmr_res.append(preds)

        vr_scores = torch.cat(vr_scores, dim=0)  # [Nq, Nv]
        vr_indices = torch.argsort(vr_scores, dim=1, descending=True).cpu().numpy().copy()

        vcmr_res_dict = dict(
            VCMR=vcmr_res,
            video2idx = test_dataset.vid_name_to_id
        )


        return vr_indices, vcmr_res_dict