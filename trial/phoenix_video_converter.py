import glob
import cv2
import os
import numpy as np

main_path = '/scratch1/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/'
video_path = os.path.join(main_path, 'features', 'fullFrame-210x260px/train')
output_path = os.path.join(main_path, 'features', 'videos', 'train')
videos = os.listdir(video_path)
for video_file in videos:
    video_output_file = output_path + '/' + video_file + '.mp4'
    if os.path.exists(video_output_file):
        continue
    img_array = []
    for filename in glob.glob(os.path.join(video_path, video_file, '*')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(video_output_file, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
