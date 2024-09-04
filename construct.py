import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from pipeline.denoiser import * 
from pipeline.denoiser.base import get_denoiser_class
from pipeline.treebuilder import *
from pipeline.capgenerator import * 
from pipeline.capgenerator import get_capgen_class
from pipeline.propgenerator import * 
from pipeline.propgenerator import get_propgen_class
from pipeline.constructpipe import * 
from pipeline.constructpipe import get_constructpipe_class

from utils.model_utils import load_pretrained_models
import utils.basic_utils as basic_utils


import warnings
warnings.filterwarnings("ignore")
from config import BuildArguments, HfArgumentParser
from dataclasses import asdict
import random 

from utils.basic_utils import seed_it


def main():
    print("Building parse pipeline")
    parser = HfArgumentParser(BuildArguments)
    cfg  = parser.parse_args_into_dataclasses(look_for_args_file=False)[0]
    print(cfg)

    seed_it(cfg.seed)

    exp_dir = os.path.join(cfg.res_dir, cfg.construct_dir, cfg.collection, cfg.construct_name)
    cfg.exp_dir = exp_dir 
    os.makedirs(exp_dir, exist_ok=True)

    cfg_dict = asdict(cfg)
    basic_utils.save_json(cfg_dict, os.path.join(exp_dir, "settings.json"))

    pretrained_models = load_pretrained_models(cfg)
    caption_generator = get_capgen_class(cfg.caption_generator)(cfg, pretrained_models)
    caption_denoiser = get_denoiser_class(cfg.caption_denoiser)(cfg, pretrained_models)
    proposal_generator = get_propgen_class(cfg.proposal_generator)(cfg, pretrained_models)

    construct_pipeline = get_constructpipe_class(cfg.construct_pipeline)(cfg, caption_generator, caption_denoiser, proposal_generator, pretrained_models)

    construct_pipeline.construct()

    print(cfg.construct_name)
    print("DONE!")

if __name__ == "__main__":
    main() 

