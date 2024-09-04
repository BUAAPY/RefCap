import torch 
from torch import Tensor
from dataset.viddataset import VideoDatasetPerSec
import os 
from PIL import Image
from torch.nn.functional import normalize

def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def normalize_min_max(t, dim):
    return (t-t.min(dim, keepdim=True)[0]) / (t.max(dim, keepdim=True)[0] - t.min(dim, keepdim=True)[0])


def _get_frame_features(
    sim_model,
    sim_processor,
    frame_images,
    cfg,
    frame_feature_path=None
):
    # dataset_pervideo = VideoDatasetPerSec(video_path, cfg.frame_resolution)
    # all_frame_images = [Image.fromarray(frame) for frame in dataset_pervideo]
    with torch.no_grad():
        # print(len(all_frame_images))
        # s = input("break")
        visual_inputs = sim_processor(images=frame_images, return_tensors="pt").to(cfg.device)
        pixel_values = visual_inputs["pixel_values"]
        vision_outputs = sim_model.vision_model(
            pixel_values=pixel_values
        )
        image_embeds = vision_outputs[0]
        frame_features = normalize(sim_model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        if frame_feature_path is not None:
            torch.save(frame_features.cpu(), frame_feature_path)
    return frame_features


def _get_caption_features(
    sim_model,
    sim_processor,
    all_frame_captions,
    cfg
):
    with torch.no_grad():
        text_inputs = sim_processor(text=all_frame_captions, return_tensors="pt", padding=True, truncation=True).to(cfg.device)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        question_embeds = sim_model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        question_embeds = question_embeds[0] 
        cap_features = normalize(sim_model.text_proj(question_embeds[:, 0, :]), dim=-1)
    return cap_features
    
    
def get_caption_frame_sims(
    sim_model,
    sim_processor,
    frame_features,
    all_frame_captions,
    cfg,
):  
    cap_features = _get_caption_features(sim_model, sim_processor, all_frame_captions, cfg)

    assert frame_features.shape == cap_features.shape 

    frame_features = frame_features.to(cap_features.device)
    ALL_SIMS = cap_features @ frame_features.t()
    sims = ALL_SIMS.diag()
    return sims, ALL_SIMS 
    
