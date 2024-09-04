from tqdm import tqdm
import os 
from pipeline.treebuilder.capTree import CapTree
import json 
import utils.basic_utils as basic_utils
import torch
import utils.sim_utils as sim_utils
import ffmpeg
from dataset.viddataset import VideoDatasetPerSec
from PIL import Image 
import spacy 


CONSTRUCTPIPE_REGISTRY = {}

def REGISTER_CONSTRUCTPIPE(names):
    def register_constructpipe_cls(cls):
        if isinstance(names, str):
            if names in CONSTRUCTPIPE_REGISTRY:
                raise ValueError(f"Cannot register duplicate construct pipeline ({names})")
            CONSTRUCTPIPE_REGISTRY[names] = cls 
        elif isinstance(names, list):
            for name in names:
                if name in CONSTRUCTPIPE_REGISTRY:
                    raise ValueError(f"Cannot register duplicate construct pipeline ({name})")
                CONSTRUCTPIPE_REGISTRY[name] = cls
        return cls 
    return register_constructpipe_cls

def get_constructpipe_class(name):
    if name not in CONSTRUCTPIPE_REGISTRY:
        raise ValueError(f"construct pipeline name {name} not registered.")
    return CONSTRUCTPIPE_REGISTRY[name]

@REGISTER_CONSTRUCTPIPE(["base"])
class BaseConstructPipeline:
    def __init__(self, cfg, caption_generator, caption_denoiser, proposal_generator, models) -> None:
        self.cfg = cfg 
        self.caption_generator = caption_generator
        self.caption_denoiser = caption_denoiser
        self.proposal_generator = proposal_generator

        vid_list = os.listdir(self.cfg.video_root)
        anno_path = os.path.join(self.cfg.anno_dir, self.cfg.collection, self.cfg.anno_file)
        # import pdb; pdb.set_trace()
        new_vid_list = self.select_videos(vid_list, anno_path)
        self.vid_list = new_vid_list
        if self.cfg.num_samples != -1:
            self.vid_list = self.vid_list[:self.cfg.num_samples] # ['xxx.mp4']
        self.it_sim_model = models['blip_itrtv_model']
        self.it_sim_processor = models['blip_itrtv_processor']

        self.create_dirs(cfg)
    
    def create_dirs(self, cfg):
        os.makedirs(cfg.meta_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.meta_dir, cfg.captions_dir), exist_ok=True)
        os.makedirs(os.path.join(cfg.meta_dir, cfg.framefeatures_dir),exist_ok=True)
        os.makedirs(os.path.join(cfg.meta_dir, cfg.raw_capframe_scores_dir),exist_ok=True)

        os.makedirs(cfg.res_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.res_dir, cfg.construct_dir), exist_ok=True)



    
    def construct(self):
        print("Generating captions...")
        captions = self.caption_generator(vid_list=self.vid_list)

        print("Computing frame features...")
        features = self.compute_frame_features(vid_list=self.vid_list, captions=captions)

        print("Computing raw capframe scores...")
        raw_scores_path = os.path.join(self.cfg.meta_dir, self.cfg.raw_capframe_scores_dir, f"{self.cfg.collection}_{self.cfg.caption_generator}.pt")
        scores = self.compute_capframe_scores(vid_list=self.vid_list, captions=captions, features=features, scores_path=raw_scores_path)

        print("Denoising captions...")
        denoised_captions = self.caption_denoiser(vid_list=self.vid_list, captions=captions, raw_scores=scores)

        print("Computing denoised capframe scores...")
        denoised_scores_path = os.path.join(self.cfg.exp_dir, self.cfg.denoised_capframe_scores_file)
        denoised_scores = self.compute_capframe_scores(vid_list=self.vid_list, captions=denoised_captions, features=features, scores_path=denoised_scores_path)

        print("Generating proposals...")
        proposals = self.proposal_generator(vid_list=self.vid_list, captions=denoised_captions, scores=denoised_scores, all_frame_features=features)

        print("Building tree_meta...")
        tree_meta = self.build_tree_meta(proposals)
        return tree_meta 
    


    def compute_capframe_scores(self, vid_list, captions, features, scores_path):
        
        if os.path.exists(scores_path):
            all_scores = torch.load(scores_path)
        else:
            all_scores = {}
        
        vid_2_cap = {x['vid_name']: x for x in captions}

        for vid in tqdm(vid_list, total=len(vid_list)):
            video_name = vid.split(".")[0]
            if video_name in all_scores:
                continue
            cap = vid_2_cap[video_name]
            all_captions = [cap["frame_captions"][k]["cap"] for k in cap["frame_captions"].keys()]
            frame_features = features[video_name]
            sims, _ = sim_utils.get_caption_frame_sims(self.it_sim_model, self.it_sim_processor, frame_features, all_captions, self.cfg)
            if self.cfg.itm_norm:
                sims = sim_utils.normalize_min_max(sims, dim=0)
            
            all_scores[video_name] = sims.cpu()
        torch.save(all_scores, scores_path)
        return all_scores



    def compute_frame_features(self, vid_list, captions, max_piece_len=100, save_freq=50):
        frame_feature_path = os.path.join(self.cfg.meta_dir, self.cfg.framefeatures_dir, f"{self.cfg.collection}.pt")
        if os.path.exists(frame_feature_path):
            all_frame_features = torch.load(frame_feature_path)
        else:
            all_frame_features = {} 

        vid_2_cap = {x['vid_name']: x for x in captions}

        new_vid_cnt = 0 

        for vid in tqdm(vid_list, total=len(vid_list)):
            video_name = vid.split(".")[0]
            if video_name in all_frame_features:
                continue
            print(video_name)
            new_vid_cnt += 1

            raw_cap = vid_2_cap[video_name]
            all_captions = [raw_cap["frame_captions"][k]["cap"] for k in raw_cap["frame_captions"].keys()]
            N = len(all_captions)
            # import pdb; pdb.set_trace()
            video_path = os.path.join(self.cfg.video_root, f"{video_name}.mp4")
            dataset_pervideo = VideoDatasetPerSec(video_path, self.cfg.frame_resolution)
            all_frame_images = [Image.fromarray(frame) for frame in dataset_pervideo]
            if N > max_piece_len:
                frame_features_list = []
                slices = []
                s = 0
                while s < N:
                    e = min(N, s+max_piece_len)
                    slices.append([s, e])
                    s = e 
                for s, e in slices:
                    frame_images = all_frame_images[s:e]
                    frame_features = sim_utils._get_frame_features(self.caption_denoiser.it_sim_model, self.caption_denoiser.it_sim_processor, frame_images, self.cfg)
                    frame_features_list.append(frame_features)
                frame_features = torch.cat(frame_features_list, dim=0)
            else:
                frame_features = sim_utils._get_frame_features(self.caption_denoiser.it_sim_model, self.caption_denoiser.it_sim_processor, all_frame_images, self.cfg)

            all_frame_features[video_name] = frame_features.cpu()
            if new_vid_cnt >= save_freq:
                torch.save(all_frame_features, frame_feature_path)
                new_vid_cnt = 0 
        
        torch.save(all_frame_features, frame_feature_path)
        return all_frame_features

    def build_tree_meta(self, proposals):
        tree_meta = {}
        for vid, metas in tqdm(proposals.items()):
            duration = metas['duration']
            proposals = metas['proposals']
            # root node 
            tree_meta[vid] = {'vid_name': vid, 'subs': [], 'caps': [], 'keys': [], 'st': 0.0, 'ed': duration, 'duration': duration}
            for prop in proposals:
                son = {'vid_name': vid, 'subs': [], 'caps': [prop['cap']], 'keys': [], 'st': prop['st'], 'ed': prop['ed'], 'duration': prop['ed']-prop['st']}
                if 'keys' in prop:
                    son['keys'] = prop['keys']
                tree_meta[vid]['subs'].append(son)
        basic_utils.save_json(tree_meta, os.path.join(self.cfg.exp_dir, self.cfg.tree_file))
        return tree_meta

    def select_videos(self, vid_list, anno_path):
        vid_names = list(map(lambda x: x.split(".")[0], vid_list))
        new_vid_list = [] 
        with open(anno_path, 'r') as f:
            all_lines = f.readlines()
            for line in all_lines:
                data = json.loads(line)
                vid_name = data['vid_name']
                if vid_name in vid_names:
                    vid = vid_list[vid_names.index(vid_name)] # xxx.mp4 or xxx.mkv
                    if vid.split(".")[-1] == "mkv":
                        mkv_video_path = os.path.join(self.cfg.video_root, vid)
                        new_vid = f"{vid_name}.mp4"
                        new_video_path = os.path.join(self.cfg.video_root, new_vid)
                        if not os.path.exists(new_video_path):
                            ffmpeg.input(mkv_video_path).output(new_video_path).run()
                        vid = new_vid
                    if vid not in new_vid_list:
                        new_vid_list.append(vid)
        print("VID CNT: ", len(new_vid_list))
        return new_vid_list