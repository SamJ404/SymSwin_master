import os
import argparse
from glob import glob

parser = argparse.ArgumentParser()
# parser.add_argument('--datasets_dir', type=list, default=['/home/wy/srdata/DIV2K/DIV2K_train_HR', '/home/wy/srdata/Flickr2K/Flickr2K_HR'])
parser.add_argument('--hrsets_dir', type=list, default=['/home/wy/srdata/NWPU_RESISC45/HR'])
parser.add_argument('--lrsets_dir', type=list, default=['/home/wy/srdata/NWPU_RESISC45/LR'])
parser.add_argument('--save_dir', type=str, default='/home/wy/dj/data/meta_info')
parser.add_argument('--txt_name', type=str, default='meta_info_RESISC45_testhr.txt')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
txt_path = os.path.join(args.save_dir, args.txt_name)
txt_file = open(txt_path, 'w')
# for dataset_dir in args.datasets_dir:
#     img_paths = sorted(glob(os.path.join(dataset_dir, '*.png')))
#     for img_path in img_paths:
#         txt_file.write(f'{img_path}\n')
for lrset_dir, hrset_dir in zip(args.lrsets_dir, args.hrsets_dir):
    for cls in sorted(os.listdir(lrset_dir)):
        lr_paths = sorted(glob(os.path.join(lrset_dir, cls, '*.png')))
        hr_paths = sorted(glob(os.path.join(hrset_dir, cls, '*.png')))
        for i, lr_path in enumerate(lr_paths):
            if i%100 == 0 or i==364 or i==634:
                txt = f'{hr_paths[i]}\n'
                txt_file.write(txt)
txt_file.close()




