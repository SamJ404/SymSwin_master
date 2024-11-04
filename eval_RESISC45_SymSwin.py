# Modified by SAMJ
# based on Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN)


from __future__ import print_function
import argparse
import os
import torch
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from collections import defaultdict
import json
from utils.utils_image import calculate_psnr
from utils.utils_image import calculate_ssim
import numpy as np
import time
from PIL import Image

from models.SymSwin.SymSwin import SymSwin as net

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--metainfo_dir', type=str, default='./data/meta_info/meta_info_RESISC_testpair.txt')
parser.add_argument('--model_type', type=str, default='SymSwin')
parser.add_argument('--pretrained_sr', default='./checkpoints/SymSwin_resisc.pth', help='sr pretrained base model')
parser.add_argument('--save_folder', default='./prediction', help='parent folder to save prediction')
parser.add_argument('--folder_name', default='SymSwin_resisc', help='folder name to save prediction')

opt = parser.parse_args()
print(opt)


def eval(opt):

    performance = defaultdict(dict)
    performance['PSNR'] = {}
    performance['SSIM'] = {}
    performance['PSNR_AVG'] = {}
    performance['SSIM_AVG'] = {}
    performance['RESISC45_psnr'] = 0
    performance['RESISC45_ssim'] = 0

    trans = transform.Compose([transform.ToTensor(), ])
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.seed)
    print('===> Building model ', opt.model_type)
    model = net()
    print('---------- Networks architecture -------------')
    model = model.to('cuda:0')
    checkpoint = opt.pretrained_sr
    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
        print('Pre-trained SR model is loaded.')
    else:
        print('No pre-trained model!!!!')
    model.eval()
    print('===> Loading val datasets')
    lr_dirs = []
    gt_dirs = []
    with open(opt.metainfo_dir, 'r') as metainfo:
        for line in metainfo.readlines():
            lr_dirs.append(line.strip().split(',')[0])
            gt_dirs.append(line.strip().split(',')[1])
    data_pre = lr_dirs[0].split('/')[-2]
    for lr_dir, gt_dir in zip(lr_dirs, gt_dirs):
        data = lr_dir.split('/')[-2]

        if not (data in performance['PSNR']):
            print(f'loading {data}')
            performance['PSNR'][data] = []
            performance['SSIM'][data] = []

        lr = Image.open(lr_dir).convert('RGB')
        lr = trans(lr).unsqueeze(0).to('cuda:0')

        with torch.no_grad():
            t0 = time.time()
            prediction = model(lr)
            t1 = time.time()

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)
        prediction = prediction.transpose(1, 2, 0)

        print("===> Processing image: %s || Timer: %.4f sec." % (lr_dir, (t1 - t0)))
        img_name, img_form = os.path.splitext(os.path.basename(lr_dir))
        sr_name = f'{img_name}_{opt.model_type}_x{opt.upscale_factor}{img_form}'
        save_foler = os.path.join(opt.save_folder, opt.folder_name, data)
        if not os.path.exists(save_foler):
            os.makedirs(save_foler)
        save_dir = os.path.join(save_foler, sr_name)
        Image.fromarray(np.uint8(prediction)).save(save_dir)

        hr = cv2.imread(gt_dir)
        pred = cv2.imread(save_dir)
        psnr_value = calculate_psnr(hr, pred, border=4)
        performance['PSNR'][data].append(psnr_value)
        ssim_value = calculate_ssim(hr, pred, border=4)
        performance['SSIM'][data].append(ssim_value)

        print('save image to:', save_dir, ' PSNR=',psnr_value, ' SSIM=', ssim_value)


        if data != data_pre and data_pre:
            performance['PSNR_AVG'][data_pre] = sum(performance['PSNR'][data_pre]) / len(performance['PSNR'][data_pre])
            performance['SSIM_AVG'][data_pre] = sum(performance['SSIM'][data_pre]) / len(performance['SSIM'][data_pre])
            print('{} average PSNR ===> {} average SSIM ===> {}'.format(data_pre, performance['PSNR_AVG'][data_pre], performance['SSIM_AVG'][data_pre]))
            performance['RESISC45_psnr'] += performance['PSNR_AVG'][data_pre]
            performance['RESISC45_ssim'] += performance['SSIM_AVG'][data_pre]
            data_pre = data
    performance['PSNR_AVG'][data] = sum(performance['PSNR'][data]) / len(performance['PSNR'][data])
    performance['SSIM_AVG'][data] = sum(performance['SSIM'][data]) / len(performance['SSIM'][data])
    print('{} average PSNR ===> {} average SSIM ===> {}'.format(data, performance['PSNR_AVG'][data],
                                                                performance['SSIM_AVG'][data]))
    performance['RESISC45_psnr'] += performance['PSNR_AVG'][data]
    performance['RESISC45_ssim'] += performance['SSIM_AVG'][data]

    performance['RESISC45_psnr'] = performance['RESISC45_psnr']/45
    performance['RESISC45_ssim'] = performance['RESISC45_ssim']/45

    print('RESISC45 average PSNR ===> {} average SSIM ===> {}'.format(performance['RESISC45_psnr'], performance['RESISC45_ssim']))

    json_dir = os.path.join(opt.save_folder, opt.folder_name, f'{opt.folder_name}.json')
    with open(json_dir, 'w') as j:
        json.dump(performance, j, indent=True, sort_keys=False, skipkeys=True)




if __name__ == '__main__':

    eval(opt=opt)