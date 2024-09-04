from dataclasses import dataclass, field

@dataclass
class BasicArguments:
    anno_dir: str = "annos"
    anno_file: str = "vcmr.jsonl"

    # below meta_dir, compute only once
    meta_dir: str = "meta"
    captions_dir: str = "captions"
    framefeatures_dir: str = "framefeatures"
    raw_capframe_scores_dir: str = "scores"

    res_dir: str = "results"
    # below construct dir, compute per construction
    construct_dir: str = "construct"
    denoised_captions_file: str = "denoised_captions.jsonl"
    denoised_capframe_scores_file: str = "denoised_capframe_scores.pt"
    prop_sim_path: str = "prop_sims.pt"
    proposals_file: str = "proposals.json"
    tree_file: str = "tree.json"
    
    retrieve_dir: str = "retrieve"


    stage: str = field(
        default="construct",
        metadata={"choices": ["construct", "retrieve"]}
    )
    collection: str = field(
        default="charardes", 
        metadata={"choices": ["charades", "activitynet"]}
    )

    video_root: str= field(
        default="/root/Charades_v1/",
        metadata={"help": "direction containing raw videos"}
    )

    device: str = "cuda"
    num_workers: int = 4
    num_samples: int = field(default=-1, metadata={"help": "-1 for all samples"})
    seed: int = 42
    max_key_cnt_per_proposal: int = 50

    # models
    caption_model: str = "blip-image-captioning-large"
    blip_itm_model: str = "blip-itm-base-coco"
    sentence_transformer: str = "paraphrase-distilroberta-v2"
    glove_model: str = "meta/glove.6B/glove.6B.300d.txt"


@dataclass
class BuildArguments(BasicArguments):
    frame_resolution: int = 384
    min_prop_size: int = 3
    construct_name: str = "construct"
    
    construct_pipeline: str = field(
        default="base",
        metadata={"choices": ["base"]}
    )
    caption_generator: str = field(
        default="minigpt",
        metadata={"choices": ["minigpt", "blip"]}
    )
    caption_denoiser: str = field(
        default="window",
        metadata={"choices": ["base", "window"]}
    )
    denoise_window_width: int = field(
        default=5,
    )
    proposal_generator: str = field(
        default="qm",
        metadata={"choices": ["qm"]}
    )
    figsim_denoise_thr: float = field(
        default=0.4,
        metadata={"help": "threshold used in figsim captioin denoising"}
    )
    itm_norm: bool = field(
        default=True,
        metadata={"help": "normalize the image-text matching scores"}
    )
    prop_sim_type: str = field(
        default="it",
        metadata={
            "help": "similarity type used in proposal generation, txt for pure text similarities, it for similarities multipled with blip itm-scores",
            "choices": ["it", "txt", "vis"]
        }
    )
    prop_score_thr: float = field(
        default=0.5,
        metadata={"help": "score threshold used in proposal generation"}
    )
    prop_kernel_width: int = field(
        default=5,
        metadata={"help": "kernel_size = 2*width+1"}
    )
    prop_min_cnt: int = 2
    prop_max_cnt: int = 5


@dataclass
class TestArguments(BasicArguments):
    construct_name: str=field(
        default="",
        metadata={"help": "$res_dir/$construct_dir/$collection/$construct_name/tree.json dir containing tree meta to be tested"}
    )
    retrieve_name: str = field(
        default="eval",
        metadata={"help": "exp name in eval"}
    )
    retrieve_pipeline: str = field(
        default="mix",
        metadata={"choices": ["sent", "key", "mix"]}
    )
    key_policy: str = field(
        default="max_mean",
        metadata={"choices": ["max_mean", "max_max"]}
    )
    eval_query_bsz: int = 200
    max_vcmr_props: int = 1000
    retrieve_sent_ratio: float = field(
        default=0.5,
        metadata={"help": "ration to balance sentence and key similarities in the retrieval process"}
    )