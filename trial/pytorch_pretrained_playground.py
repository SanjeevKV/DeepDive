import img2vec as iv
import os
import torchvision.models as models
import torch

def get_files_from_dir(dir_path):
    return sorted([x for x in os.listdir(dir_path) if '.png' in x ])

if __name__ == "__main__":
    dir_path = '/scratch2/maiyaupp/how2sign/train_images/3Uu9_GGILmc_23-5-rgb_front/'
    files = get_files_from_dir(dir_path)
    vid = iv.prepare_video(files, dir_path)
    #model = models.alexnet(pretrained=True)
    #model.classifier = model.classifier[:6]
    model = models.resnet101(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    out = model(vid)
    print(len(vid), type(vid), len(out), type(out))
    print(vid.shape, out.shape)
    print(torch.count_nonzero(out))
    print(torch.count_nonzero(out) / (out.shape[0] * out.shape[1]))