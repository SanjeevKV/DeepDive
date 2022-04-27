from cv2 import imread, resize
import numpy as np
import os
import gzip
import pickle
import argparse
import sys
import torch
import torchvision.models as models
from torchvision import transforms #11August_2010_Wednesday_tagesschau-1
from img2vec import prepare_image, prepare_video
from torch.autograd import Variable
from pytorchcv.model_provider import get_model

# from mmpose.apis.inference import init_pose_model

from PIL import Image

video_path = '/scratch1/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train/31May_2011_Tuesday_tagesschau-4301'#11August_2010_Wednesday_tagesschau-1' #21October_2010_Thursday_tagesschau-1816'#
anno = ['11August_2010_Wednesday_tagesschau-1', '11August_2010_Wednesday_tagesschau-2/1/*.png', '-1', '-1', 'Signer08', 'DRUCK TIEF KOMMEN', 'tiefer luftdruck bestimmt in den nächsten tagen unser wetter']
im1_path = '/scratch1/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train/11August_2010_Wednesday_tagesschau-1/images0001.png'

# input_image = Image.open(im1_path)
# img = transforms.ToTensor()(input_image)

# input_image = Image.open(im1_path)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# img = preprocess(input_image)
#this sample code from alexnet also ends up in the -1.8044 to 2.1975 range
def try_model(model):
    files = sorted([x for x in os.listdir(video_path) if '.png' in x ])
    vid = prepare_video(files, video_path)
    print(vid.min(), vid.max())
    print(vid.shape)
    with torch.no_grad():
        out = model(vid)
    print(out.count_nonzero(), out.shape, out.numel(), out.min(), out.max())
    return out
if __name__=='__main__':
    # cfg = '/home1/ssnair/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
    # chkpnt = '/home1/ssnair/mmpose/my_work/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
    # model = init_pose_model(cfg, checkpoint=chkpnt, device='cpu')
    # model.backbone.stage4[2].relu = torch.nn.Identity()
    # model = model.backbone
    # out = try_model(model)
    model = get_model("alphapose_fastseresnet101b_coco", pretrained=True)
    model.decoder = torch.nn.Identity()
    model.backbone.stage4.unit3.activ = torch.nn.Identity()
    model.heatmap_max_det = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Flatten()
        )
    out = try_model(model)

# model = models.alexnet(pretrained=True)
# model = models.vit_b_16(pretrained=True)
# out = try_model(model)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# model.classifier = model.classifier[:2]
# print(img, img.shape, img.min(), img.max())

#without Normalize ranges from 0 to 255
#with Normalize ranges from -2.1179 to 1113.7511
#with Normalize and img/255 ranges from -2.1179 to 2.5758







# from alexnet_tf2 import AlexNet
# from cv2 import imread, resize
# import numpy as np
# import torch
# import torchvision.models as models

# def prepare_image(fp):
# 	img = imread(fp)
# 	img = resize(img, (227, 227))
# 	img = img.reshape(1, *img.shape)
# 	img = img.astype(np.float32)
# 	img = torch.tensor(img)
# 	img = torch.transpose(img, 1, 3)
# 	assert img.shape == (1, 3, 227, 227)
# 	return img
# #device = "/gpu:0"
# device = 'cpu'
# keep_rate = 0.8
# pa = '/scratch2/ssnair/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev/'
# filepath = pa + '11August_2010_Wednesday_tagesschau-8/'
# img1 = prepare_image(filepath + 'images0001.png')
# # cnn = AlexNet(img1, keep_rate, device)
# img2 = prepare_image(filepath + 'images0002.png')
# model = models.alexnet(pretrained=True)
# model.classifier = model.classifier[:5]
# img = torch.cat([img1, img2], dim = 0)
# print(img.shape)
# out = model(img)
# print(out.shape)
# #using tensorflow 1.3.0
