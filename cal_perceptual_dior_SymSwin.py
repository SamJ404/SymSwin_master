# modified by SAMJ
# based on CLIPSCORE (https://github.com/satlas-super-resolution/tree/main

import clip
import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import lpips
import cv2
import os
from glob import glob
from collections import defaultdict
import json
import argparse


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPSCORE(nn.Module):
    def __init__(self, clip_model, loss_weight):
        super().__init__()
        self.device = torch.device('cuda')
        self.loss_weight = loss_weight

        if clip_model == 'clip-ViT-B/16':
            self.model, _ = clip.load("ViT-B/16", device=self.device)
            self.img_size = (224, 224)
        elif clip_model == 'clipa-ViT-bigG-14':
            model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
            self.model = model.to(self.device)
            self.img_size = (336, 336)
        elif clip_model == 'siglip-ViT-SO400M-14':
            model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
            self.model = model.to(self.device)
            self.img_size = (384, 384)
        # else:
        #     print(clip_model, " is not supported for CLIPScore.")
        else:
            self.model, _ = clip.load(clip_model, device=self.device)
            self.img_size = (224, 224)

        self.normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)


    def forward(self, x, gt, **kwargs):

        tensor1 = torch.as_tensor(x).permute(2, 0, 1)
        tensor1 = tensor1.unsqueeze(0).to(self.device).float()/255
        tensor2 = torch.as_tensor(gt).permute(2, 0, 1)
        tensor2 = tensor2.unsqueeze(0).to(self.device).float()/255

        tensor1 = F.interpolate(tensor1, self.img_size)
        tensor2 = F.interpolate(tensor2, self.img_size)

        feats1 = self.model.encode_image(tensor1)
        feats2 = self.model.encode_image(tensor2)

        clip_score = F.cosine_similarity(feats1, feats2).detach().item()

        return clip_score


class LPIPS_score(nn.Module):
    def __init__(self, net='vgg'):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = lpips.LPIPS(net=net).to(self.device)

    def forward(self, pred, gt):

        tensor1 = torch.as_tensor(pred).permute(2, 0, 1)
        tensor1 = tensor1.unsqueeze(0).to(self.device).float()/255
        tensor2 = torch.as_tensor(gt).permute(2, 0, 1)
        tensor2 = tensor2.unsqueeze(0).to(self.device).float()/255

        loss = self.model(tensor1, tensor2).detach().item()
        return loss



if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Perceptual metric calculation')
    parser.add_argument('--clip_model', type=str, default='./pretrained_models/ViT-B-16.pt', help="super resolution upscale factor")
    parser.add_argument('--metainfo_dir', type=str, default='./data/meta_info/meta_info_resisc_testpair.txt')
    parser.add_argument('--predset', type=str, default='./prediction/SymSwin_resisc')

    opt = parser.parse_args()

    clipscore = CLIPSCORE(opt.clip_model, 0)
    lpips = LPIPS_score('vgg')
    performance = defaultdict(dict)
    gt_dirs = []
    with open(opt.metainfo_dir,'r') as gt_file:
        for i,gt_dir in enumerate(gt_file.readlines()):
            if i%100 == 0:
                gt_dirs.append(gt_dir.strip('\n').split(',')[1])
    gt_file.close()
    sr_dirs = sorted(glob(os.path.join(opt.predset,'*.jpg')))
    for gt_dir, sr_dir in zip(gt_dirs, sr_dirs):
        gt = cv2.imread(gt_dir)
        sr = cv2.imread(sr_dir)
        performance['clipscore'][sr_dir.split('/')[-1]] = clipscore(sr, gt)
        performance['lpips'][sr_dir.split('/')[-1]] = lpips(sr, gt)

    performance['avg_clipscore'] = sum(performance['clipscore'].values())/len(performance['clipscore'])
    performance['avg_lpips'] = sum(performance['lpips'].values()) / len(performance['lpips'])
    print('avg_clipscore={}'.format(performance['avg_clipscore']))
    print('avg_lpips={}'.format(performance['avg_lpips']))
    with open(os.path.join(opt.predset, 'perceptural_score.json'), 'w') as j:
        json.dump(performance, j, indent=True, sort_keys=False, skipkeys=True)

