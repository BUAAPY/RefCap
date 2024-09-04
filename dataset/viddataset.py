import numpy as np
import torch
from torch.utils.data import Dataset
import ffmpeg
import os
from PIL import Image


class VideoDatasetPerSec(Dataset):
    """Pytorch video loader."""
    def __init__(self, video_path, size=384):
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
            # video = torch.from_numpy(video.astype('float32'))
            # video = video.permute(0, 3, 1, 2)
        else:
            print('file not find: {}'.format(video_path))
            video = np.zeros(1)
            
        return video

    def __getitem__(self, idx):
        return self.frames[idx]