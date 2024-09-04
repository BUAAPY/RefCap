# Introduction

Official implementation of paper:

RefCap: Zero-shot Video Corpus Moment Retrieval Based on Refined Dense Video Captioning



# Environment Setup

```
conda create -n refcap python=3.10
conda activate refcap

apt-get update
apt-get install ffmpeg

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

* follow [Glove](https://github.com/stanfordnlp/GloVe) to prepare glove weights, and replace them in `meta` folder:

  * ```
    meta
    |--glove.6B
    |  |--glove.6B.300d.txt
    ```

# Construction Stage

## BLIP as VLLM

* ```
  #Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
  source setup.sh
  
  # set ${video_root}, ${collection} correctly in scripts/construct.sh
  # set ${caption_generator} to 'blip' in scripts/construct.sh 
  bash scripts/construct.sh
  ```

  * Note: for ActivityNet dataset, please merge all videos from  split `v1-2` and `v1-3` and place them in one folder.

* generated intermediate results will be placed in `meta` folder (generated **only once** for specified `collection `and `caption_generator`):

  * ```
    meta
    |--captions							// containing generated frame captions
    |  |--${collection}_${caption_generator}.jsonl
    |--framefeatures  					// containing extracted blip frame features
    |  |--${collection}.pt
    |--scores 							// containing caption-frame similarity scores before denoising
    |  |--${collection}_{caption_generator}.pt
    |--glove.6B
    |  |--glove.6B.300d.txt
    ```

  * Our generated intermediate results are provided in [Baidu Cloud Disk]( https://pan.baidu.com/s/1Sn41an8cpd9qbKMTQX2GkA?pwd=tmi3 )  and [Google Drive](https://drive.google.com/drive/folders/1HI2jraauWR_ilAJC5N7y0gk1rvY1w5fM?usp=drive_link)

* generated construction results will be placed in `results/construct/${collection}/${construct_name}`(per construction)

  * ```
    results
    |--construct
    |  |--${collection}
    |  |  |--${construct_name}
    |  |  |  |--settings.json
    |  |  |  |--denoised_capframe_scores.pt
    |  |  |  |--denoised_captions.jsonl
    |  |  |  |--prop_sims.pt
    |  |  |  |--proposals.json
    |  |  |  |--tree.json
    ```

## MiniGPT as VLLM

* For MiniGPT, please follow the [official guidance](https://github.com/Vision-CAIR/MiniGPT-4) to download MiniGPT-4, prepare environment and weights. Then place our provided script `utils/genCaptions_minigpt.py` in the root directory of MiniGPT-4, and run the script as:

  * ```
    python genCaptions_minigpt.py --collection $collection --save_dir $save_dir --video_root $video_root --split_json_path $split_json_path --temperature $temperature
    ```

  * Generated captions will be saved in `$save_dir/${collection}_minigpt.jsonl`
  * Intermediate results are also provided in  [Baidu Cloud Disk]( https://pan.baidu.com/s/1Sn41an8cpd9qbKMTQX2GkA?pwd=tmi3 )  and [Google Drive](https://drive.google.com/drive/folders/1HI2jraauWR_ilAJC5N7y0gk1rvY1w5fM?usp=drive_link)

* Other operations are the same as BLIP:

  * ```
    source setup.sh
    
    # set ${video_root}, ${collection} correctly in scripts/construct.sh
    # set ${caption_generator} to 'minigpt' in scripts/construct.sh 
    bash scripts/construct.sh
    ```



# Retrieval Stage

* ```
  source setup.sh
  
  # set ${collection}, ${construct_name}, ${retrieve_name} correctly in scripts/retrieve.sh
  # construction results in results/construct/${collection}/${construct_name} will be used for retrieval
  bash scripts/retrieve.sh
  ```

* retrieval results will be placed in `results/retrieve/${collection}/${retrieve_name}`:

  * ```
    results
    |--construct  // containing construct results
    |--retrieve
    |  |--${collection}
    |  |  |--${retrieve_name}
    |  |  |  |--build_settings.json
    |  |  |  |--eval_settings.json
    |  |  |  |--metrics.json
    |  |  |  |--vcmr_preds.json
    ```

    

* We provide construction and retrieval results for ActivityNet/Charades datasets and Blip/MiniGPT VLLMs in [Baidu Cloud Disk]( https://pan.baidu.com/s/1Sn41an8cpd9qbKMTQX2GkA?pwd=tmi3 )  and [Google Drive](https://drive.google.com/drive/folders/1HI2jraauWR_ilAJC5N7y0gk1rvY1w5fM?usp=drive_link).