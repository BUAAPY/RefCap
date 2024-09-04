import torch.utils.data as data
from utils.basic_utils import load_jsonl
import copy 



class DataSet4Test(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, captree_meta=None):
        # Captions
        self.desc_names = []  
        self.desc_ids = [] 
        self.descs = [] 

        self.vid_names = []
        self.vid_ids = [] 
        self.vid_name_to_id = {} 
        self.vid_id_to_name = {} 

        self.gt_descid_to_vidid = {}  # {desc_id: [gt_vid_id1,...]}
        self.desc_id_to_anno = {}
        self.time_stamps = [] 
        
        cap_data = load_jsonl(cap_file)
        print("test num: ", len(cap_data))
        if captree_meta is not None:
            part_cap_data = [] 

        vid_unique_list = []
        desc_id = 0 

        

        for item in cap_data:
            if captree_meta is not None and item['vid_name'] not in captree_meta:
                continue
            item['desc_id'] = desc_id
            desc_id += 1
            vid_name = item['vid_name']
            if vid_name in vid_unique_list:
                item['vid_id'] = vid_unique_list.index(vid_name)
            else:
                item['vid_id'] = len(vid_unique_list)
                vid_unique_list.append(vid_name) 

            self.desc_names.append(item['desc_name'])
            self.desc_ids.append(item['desc_id'])
            self.descs.append(item['desc'])
            self.vid_names.append(item['vid_name'])
            self.vid_ids.append(item['vid_id'])
            self.time_stamps.append(item["ts"])
            self.vid_name_to_id[item['vid_name']] = item['vid_id']
            self.vid_id_to_name[item['vid_id']] = item['vid_name']

            if item['desc_id'] not in self.gt_descid_to_vidid:
                self.gt_descid_to_vidid[item['desc_id']] = [item['vid_id']]
            else:
                self.gt_descid_to_vidid[item['desc_id']].append(item['vid_id'])
            
            self.desc_id_to_anno[item['desc_id']] = item 
            
            if captree_meta is not None:
                part_cap_data.append(copy.deepcopy(item))
        
        if captree_meta is not None:
            self.cap_data = part_cap_data
        else:
            self.cap_data = cap_data


    def __getitem__(self, index):
        # str, str, str 
        desc_name, desc_id, vid_name, vid_id, desc, ts = self.desc_names[index], self.desc_ids[index], self.vid_names[index], self.vid_ids[index], self.descs[index], self.time_stamps[index]
        return desc_name, desc_id, vid_name, vid_id, desc, ts[0], ts[1]

    def __len__(self):
        return len(self.cap_data)




