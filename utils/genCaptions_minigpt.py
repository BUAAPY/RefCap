import numpy as np
import torch
from torch.utils.data import Dataset
import ffmpeg
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from PIL import Image
import random 
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
import json

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class VideoDatasetPerSec(Dataset):
    """Pytorch video loader."""
    def __init__(self, video_path, size=384 ):
        self.size = size
        self.frames = self._get_video_frames(video_path)

    def __len__(self):
        return len(self.frames)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        duration = float(video_stream['duration'])
        self.duration = duration
        return height, width, duration

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def _get_video_frames(self, video_path):
        if os.path.isfile(video_path):
            # print('Decoding video: {}'.format(video_path))
            try:
                h, w, duration = self._get_video_dim(video_path)
            except Exception as e:
                print(e)
                print('ffprobe failed at: {}'.format(video_path))
                return torch.zeros(1)
            height, width = self._get_output_dim(h, w)
            frames = [] 
            for i in range(int(duration)):
                cmd = (
                    ffmpeg
                    .input(video_path, ss=i, t=1)
                    .filter('scale', width, height)
                )
                out, _ = (
                    cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
                img = Image.frombytes('RGB', (width, height), out)
                img_np = np.array(img)
                frames.append(img_np)
            video = np.stack(frames, axis=0)
        else:
            print('file not find: {}'.format(video_path))
            video = np.zeros(1)
            
        return video

    def __getitem__(self, idx):
        return self.frames[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="[0,1], higher more flexible"
    )

    # 
    parser.add_argument("--collection", required=True, type=str, choices=["activitynet", "charades"])
    parser.add_argument('--save_dir', default='/root/RefCap/meta/captions', type=str)
    parser.add_argument('--video_root', default='/root/Charades_v1/', type=str)
    parser.add_argument('--split_json_path', type=str, default='/root/RefCap/annos/charades/vcmr.jsonl')
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--input_size', default=384, type=int)
    
    args = parser.parse_args()
    return args


def gpt_generate_caption(image, chat, user_message):
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image, chat_state, img_list)

    chat.ask(user_message, chat_state)
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)

    streamer = chat.stream_answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=args.temperature,
                                  max_new_tokens=500,
                                  max_length=2000)
    caption = ""
    for output in streamer:
        caption = caption + output
    
    return caption 


def generate_caption(args, chat, prompt, video_name, video_path, captions_save_path):
    
    dataset_pervideo = VideoDatasetPerSec(video_path, args.input_size)
    duration = dataset_pervideo.duration
    if len(dataset_pervideo.frames.shape) != 4:
        return 
    res = {'vid_name': video_name, 'frame_captions': {}, 'duration': duration} 
    for id, frame in tqdm(enumerate(dataset_pervideo), total=len(dataset_pervideo)):
        image_pil = Image.fromarray(frame).convert('RGB')
        cap = gpt_generate_caption(image_pil, chat, user_message=prompt)
        res['frame_captions'][id] = {
            "cap": cap.lower(),
        }
    
    with open(captions_save_path, "a+") as f:
        res_s = json.dumps(res)
        f.writelines(f"{res_s}\n")
    
def main(args):
    # Build Minigpt-v
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}")
    args.device = device
    cfg = Config(args)
    device = 'cuda:{}'.format(args.gpu_id)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    model = model.eval()
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    ### Change Prompt Here!
    prompt = 'Please describe this image with at most 30 words'

    # Load Video Paths
    os.makedirs(args.save_dir, exist_ok=True)
    vid_list = os.listdir(args.video_root)
    vid_names = list(map(lambda x: x.split(".")[0], vid_list))

    new_vid_list = [] 
    with open(args.split_json_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            data = json.loads(line)
            vid_name = data['vid_name']
            if vid_name in vid_names:
                vid = vid_list[vid_names.index(vid_name)]
                if vid not in new_vid_list:
                    new_vid_list.append(vid)
    vid_list = new_vid_list
    if args.num_samples != -1:
        vid_list = vid_list[:args.num_samples] # ['xxx.mp4']
    # import pdb; pdb.set_trace()

    captions_save_path = os.path.join(cfg.save_dir, f"{cfg.collection}_minigpt.jsonl")
    if os.path.exists(captions_save_path):
        captions = load_jsonl(captions_save_path)
        already_video_names = set([x['vid_name'] for x in captions])
    else:
        captions = []
        already_video_names = set() 

    for vid in tqdm(vid_list,total=len(vid_list)):
        video_name = vid.split('.')[0]
        
        if vid.split(".")[-1] == "mkv": # convert mkv to mp4
            # import pdb; pdb.set_trace()
            mkv_video_path = os.path.join(args.video_root, vid)
            new_vid = f"{vid_name}.mp4"
            new_video_path = os.path.join(args.video_root, new_vid)
            if not os.path.exists(new_video_path):
                ffmpeg.input(mkv_video_path).output(new_video_path).run()
            vid = new_vid
        video_path = os.path.join(args.video_root, vid)
        print(vid)
        if not os.path.isfile(video_path):
            print("Video Not Found!")
            continue
        if video_name in already_video_names:
            continue
        generate_caption(args=args,chat=chat, prompt=prompt, video_name=video_name, video_path=video_path, captions_save_path=captions_save_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)



