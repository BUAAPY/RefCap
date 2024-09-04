
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval, BlipModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from torchtext.vocab import Vectors
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch 

def load_pretrained_models(cfg):
    if cfg.stage == "construct":
        if cfg.caption_generator == "blip":
            print("load cap_gen_model::blip")
            cap_gen_model = BlipForConditionalGeneration.from_pretrained(cfg.caption_model).to(cfg.device)
            cap_gen_processor = BlipProcessor.from_pretrained(cfg.caption_model)
        elif cfg.caption_generator == "minigpt":
            print("Minigpt capgen:: You should generate captions using external scripts")
            cap_gen_model = None 
            cap_gen_processor = None 
        elif cfg.caption_generator == "llava":
            print("load cap_gen_model::llava")
            cap_gen_model = LlavaForConditionalGeneration.from_pretrained(
                cfg.caption_model,
                torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(cfg.device)
            cap_gen_processor = AutoProcessor.from_pretrained(cfg.caption_model)
    else:
        cap_gen_model = None 
        cap_gen_processor = None 

    print("load blip_retrieval_model")
    blip_itrtv_model = BlipForImageTextRetrieval.from_pretrained(cfg.blip_itm_model).to(cfg.device)
    blip_itrtv_processor = BlipProcessor.from_pretrained(cfg.blip_itm_model)

    
    print("load sentence transformer & keywords model")
    sentence_transformer = SentenceTransformer(cfg.sentence_transformer).to(cfg.device)
    glove_model = Vectors(name=cfg.glove_model, cache=cfg.meta_dir)

    return {
        "cap_gen_model": cap_gen_model,
        "cap_gen_processor": cap_gen_processor,
        "blip_itrtv_model": blip_itrtv_model,
        "blip_itrtv_processor": blip_itrtv_processor,
        "sentence_transformer": sentence_transformer,
        "glove_model": glove_model
    }