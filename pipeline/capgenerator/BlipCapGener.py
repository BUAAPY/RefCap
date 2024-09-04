from .base import REGISTER_CAPGEN, BaseCapGen
import os 
from dataset.viddataset import VideoDatasetPerSec
from tqdm import tqdm 
from PIL import Image
import utils.basic_utils as basic_utils
import json 


@REGISTER_CAPGEN(["blip"])
class CapGeneratorBLIP(BaseCapGen):
    def __init__(self, cfg, models) -> None:
        super(CapGeneratorBLIP, self).__init__(cfg, models)
        self.cap_model = models['cap_gen_model']
        self.cap_processor = models['cap_gen_processor']
    
    def generate_caption(self, video_name, video_path):
        if video_name in self.already_video_names:
            return 
        
        dataset_pervideo = VideoDatasetPerSec(video_path, self.cfg.frame_resolution)
        duration = dataset_pervideo.duration
        if len(dataset_pervideo.frames.shape) != 4:
            return 
        res = {'vid_name': video_name, 'frame_captions': {}, 'duration': duration} 
        for id, frame in tqdm(enumerate(dataset_pervideo), total=len(dataset_pervideo)):
            image_pil = Image.fromarray(frame)
            text = "a photo of"
            cap_inputs = self.cap_processor(image_pil, text, return_tensors="pt").to(self.cfg.device)
            out = self.cap_model.generate(**cap_inputs)
            cap = self.cap_processor.decode(out[0], skip_special_token=True)
            cap = cap.replace(" [SEP]", ".")
            cap = cap.replace("a photo of ", "")
            res['frame_captions'][id] = {
                "cap": cap.lower() 
            }
        
        with open(self.captions_save_file, "a+") as f:
            res_s = json.dumps(res)
            f.writelines(f"{res_s}\n")
        self.captions.append(res)
        self.already_video_names.add(video_name)
