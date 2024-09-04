import json 
import torch 
from sentence_transformers import SentenceTransformer, util
from torchtext.vocab import Vectors
import numpy as np
import argparse
from typing import * 
import copy 
import os 
from tqdm import tqdm 


class CapTree:
    def __init__(self, cfg, tree_meta, models) -> None:
        self.cfg = cfg 
        if type(tree_meta) == str:
            with open(tree_meta, 'r') as f:
                self.tree_meta = json.load(f)
        elif type(tree_meta) == dict:
            self.tree_meta = tree_meta
        
        self.cap_feature_model = models["sentence_transformer"]
        self.key_feature_model = models["glove_model"]
        self.itm_feature_model = models["blip_itrtv_model"]
        self.itm_feature_processor = models["blip_itrtv_processor"] # used for computing itm scores between given input texts and visual pooling features of each proposal

        self.node_cnt = 0 
        self.cap_cnt = 0 
        self.key_cnt = 0  
        self.root_cnt = 0

        self.max_key_cnt_per_proposal = self.cfg.max_key_cnt_per_proposal

        self.nodeid_to_node = {} # {id: node}, node: {'vid_name', "subs", "caps", "keys", "st", "ed", "duration"}
        self.caps = []
        self.keys = [] 
        self.cap_to_nodeid = []  
        self.key_to_nodeid = []  
        self.prop_cnt_per_video = [] 
        self.vidname_to_capids = {}  
        self.vidname_to_keyids = {} 
        self.vidname_to_durations = {}
        self.prop_key_cnts = [] 
        self.cap_features = None 
        self.key_features = None 

    def build_relations(self):
        qu = [] 
        node_id = 0 
        cap_id = 0 
        key_id = 0 
        
        for vid_id, root in self.tree_meta.items(): 
            root['node_id'] = node_id
            self.nodeid_to_node[root['node_id']] = root 
            self.vidname_to_durations[root['vid_name']] = root['duration']
            node_id += 1
            self.root_cnt += 1
            if len(root['subs']) > 0:
                self.prop_cnt_per_video.append(len(root['subs']))
                qu.append(root)
        
        while len(qu) > 0:
            root = qu.pop(0)
            for node in root['subs']:
                node['node_id'] = node_id
                self.nodeid_to_node[node['node_id']] = node 
                node_id += 1
                if len(node['caps']) > 0:
                    self.cap_cnt += len(node['caps'])
                    self.caps += node['caps']
                    self.cap_to_nodeid += [node['node_id']]*len(node['caps'])
                    if node['vid_name'] not in self.vidname_to_capids:
                        self.vidname_to_capids[node['vid_name']] = []
                    self.vidname_to_capids[node['vid_name']] += list(range(cap_id, cap_id+len(node['caps'])))
                    cap_id += len(node['caps'])
                if len(node['keys']) > 0:
                    node['keys'] = list(filter(lambda x: x in self.key_feature_model.stoi, node['keys']))
                    node['keys'] = node['keys'][:self.max_key_cnt_per_proposal]
                    self.key_cnt += len(node['keys'])
                    self.keys += node['keys']
                    self.key_to_nodeid += [node['node_id']]*len(node['keys'])
                    if node['vid_name'] not in self.vidname_to_keyids:
                        self.vidname_to_keyids[node['vid_name']] = []
                    self.vidname_to_keyids[node['vid_name']] += list(range(key_id, key_id+len(node['keys'])))
                    key_id += len(node['keys'])
                    self.prop_key_cnts.append(len(node['keys']))
                if len(node['subs']) > 0:
                    qu.append(node)
        self.node_cnt = node_id
        self.cap_to_nodeid = torch.tensor(self.cap_to_nodeid).to(self.cfg.device)
        self.key_to_nodeid = torch.tensor(self.key_to_nodeid).to(self.cfg.device)
        print(f"Total::  cap_cnt: {self.cap_cnt}, key_cnt: {self.key_cnt}, root_cnt: {self.root_cnt}, node_cnt: {self.node_cnt}")
    
    def encode_caps(self, caps: List[str], show_progress_bar=False):
        cap_features = self.cap_feature_model.encode(caps, show_progress_bar=show_progress_bar)
        cap_features = torch.tensor(cap_features).to(self.cfg.device)
        return cap_features
    
    def encode_keys(self, keys: List[str]):
        key_features = self.key_feature_model.vectors[[self.key_feature_model.stoi[w] for w in keys]]
        key_features = key_features.to(self.cfg.device)
        return key_features

    def compute_tree_feature(self, resume_video_names):
        new_tree_meta = {}
        for vid_name, meta in self.tree_meta.items():
            if vid_name in resume_video_names:
                new_tree_meta[vid_name] = copy.deepcopy(meta)
        self.tree_meta = new_tree_meta

        self.build_relations()
        print("compute tree features..")
        self.cap_features = self.encode_caps(self.caps, show_progress_bar=True) # [Np, E]
        self.key_features = self.encode_keys(self.keys)
        print("Done!")
    





        