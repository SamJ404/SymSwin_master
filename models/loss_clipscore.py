# Zooming out on zooming in: advancing super-resolution for remote-sensing
# (https://github.com/allenai/satlas-super-resolution/tree/main)

import clip
import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

import cv2
import os
from glob import glob
from collections import defaultdict
import json


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

        # # tensor1 = torch.squeeze(img1, dim=0)
        # # tensor2 = torch.squeeze(img2, dim=0)
        # tensor1 = torch.as_tensor(x).permute(2, 0, 1)
        # tensor1 = tensor1.unsqueeze(0).to(self.device).float()/255
        # tensor2 = torch.as_tensor(gt).permute(2, 0, 1)
        # tensor2 = tensor2.unsqueeze(0).to(self.device).float()/255
        #
        # tensor1 = F.interpolate(tensor1, self.img_size)
        # tensor2 = F.interpolate(tensor2, self.img_size)
        #
        #
        #
        # feats1 = self.model.encode_image(tensor1)
        # feats2 = self.model.encode_image(tensor2)
        #
        # clip_score = F.cosine_similarity(feats1, feats2).detach().item()
        #
        # return clip_score

        x = F.interpolate(x, self.img_size)
        gt = F.interpolate(gt, self.img_size)

        x = self.normalize(x)
        gt = self.normalize(gt)

        x_feats = self.model.encode_image(x)
        gt_feats = self.model.encode_image(gt)
        clip_score = F.l1_loss(x_feats, gt_feats, reduction='mean')

        return clip_score*self.loss_weight

if __name__ =='__main__':
    clipscore = CLIPSCORE('/ViT-B-16.pt', 0)
    performance = defaultdict(dict)
    gt_dirs = []
    with open('/data/meta_info/meta_info_RESISC45_testpair.txt','r') as gt_file:
        for gt_dir in gt_file.readlines():
            gt_dirs.append(gt_dir.strip('\n').split(',')[1])
    gt_file.close()
    sr_dirs = []
    for cls in sorted(os.listdir('predict/image/dir')):
        cls_dir = os.path.join('predict/image/dir', cls)
        sr_dirs.append(sorted(glob(os.path.join(cls_dir,'.png'))))
    sr_dirs = sum(sr_dirs, [])
    for gt_dir, sr_dir in zip(gt_dirs, sr_dirs):
        gt = cv2.imread(gt_dir)
        sr = cv2.imread(sr_dir)
        performance[gt_dir.split('/')[-2:]] = clipscore(sr, gt)
    performance['avg_clipscore'] = sum(performance.values())/len(performance)
    with open('/clipscore.json', 'w') as j:
        json.dump(performance, j, indent=True, sort_keys=False, skipkeys=True)

