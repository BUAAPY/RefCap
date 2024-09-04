#!/bin/bash
collection=charades                 # charades, activitynet
caption_generator=minigpt           # minigpt, blip
video_root=/dir/to/videos 

construct_name=minigpt_window_it
caption_denoiser=window                     # base(nodenoising), window
prop_sim_type=it                            # it, txt, vis

construct_pipeline=base 
figsim_denoise_thr=0.4
denoise_window_width=2
proposal_generator=qm     

stage=construct
prop_score_thr=0.2
prop_kernel_width=5
prop_min_cnt=2
prop_max_cnt=5
num_samples=-1
seed=42

## Model
# Optional: replace with local paths for HF models
caption_model=blip-image-captioning-large
blip_itm_model=blip-itm-base-coco
sentence_transformer=paraphrase-distilroberta-v2
glove_model=meta/glove.6B/glove.6B.300d.txt






echo $construct_name
python construct.py \
--stage $stage \
--seed $seed \
--collection $collection \
--video_root $video_root \
--construct_name $construct_name \
--num_samples $num_samples \
--figsim_denoise_thr $figsim_denoise_thr \
--prop_sim_type $prop_sim_type \
--prop_score_thr $prop_score_thr \
--prop_kernel_width $prop_kernel_width \
--prop_min_cnt $prop_min_cnt \
--prop_max_cnt $prop_max_cnt \
--caption_model $caption_model \
--blip_itm_model $blip_itm_model \
--sentence_transformer $sentence_transformer \
--glove_model $glove_model \
--construct_pipeline $construct_pipeline \
--caption_generator $caption_generator \
--caption_denoiser $caption_denoiser \
--proposal_generator $proposal_generator \
--denoise_window_width $denoise_window_width
