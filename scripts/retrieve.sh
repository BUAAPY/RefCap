#!/bin/bash
stage=retrieve
collection=charades                 # activitynet, charades

retrieve_name=minigpt_window_it_mixinfer_maxmean
construct_name=minigpt_window_it
retrieve_pipeline=mix               # sent, key, mix
key_policy=max_mean                 # max_mean, max_max

eval_query_bsz=50
max_vcmr_props=1000
retrieve_sent_ratio=0.5
seed=42

## Model
caption_model=blip-image-captioning-large
blip_itm_model=blip-itm-base-coco
sentence_transformer=paraphrase-distilroberta-v2
glove_model=meta/glove.6B/glove.6B.300d.txt

echo $retrieve_name

python retrieve.py \
--stage $stage \
--seed $seed \
--collection $collection \
--construct_name $construct_name \
--retrieve_name $retrieve_name \
--retrieve_pipeline $retrieve_pipeline \
--eval_query_bsz $eval_query_bsz \
--max_vcmr_props $max_vcmr_props \
--retrieve_sent_ratio $retrieve_sent_ratio \
--caption_model $caption_model \
--blip_itm_model $blip_itm_model \
--sentence_transformer $sentence_transformer \
--glove_model $glove_model \
--key_policy $key_policy