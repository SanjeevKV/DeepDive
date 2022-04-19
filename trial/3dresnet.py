import torch
import glob
import cv2
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


img_array = []
for filename in glob.glob('/scratch2/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/videos/frames/*'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
model.blocks[5].proj = None
model.blocks[5].output_pool = None
model = model.eval()
model = model.to(device)

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            # UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second
start_sec = 0
end_sec = start_sec + clip_duration

video = EncodedVideo.from_path('video.mp4')
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
video_data = transform(video_data)

inputs = video_data["video"]
inputs = inputs.to(device)

preds = model(inputs[None, ...])
preds = torch.nn.AdaptiveAvgPool3d(1)(preds)
preds = torch.squeeze(preds)
print(preds)
print(preds.shape)