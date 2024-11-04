# by SAM J

import argparse
import glob
import os
import cv2


def main(args):
    txt_file = open(args.meta_info, 'w')
    # RESISC
    img_paths_gt = []
    img_paths_lq = []
    # for cls in sorted(os.listdir(args.input_gt)):
    #     hrcls_dir = os.path.join(args.input_gt, cls)
    #     lrcls_dir = os.path.join(args.input_lr, cls)
    #     os.makedirs(lrcls_dir, exist_ok=True)
    #     hr_dirs = sorted(glob.glob(os.path.join(hrcls_dir, '*.png')))
    #     for i,hr_dir in enumerate(hr_dirs):
    #         if i == 364 or i == 634 or i%100 == 0:
    #             img_paths_gt.append(hr_dir)
    #             hr_nm = os.path.basename(hr_dir)
    #             hr = cv2.imread(hr_dir)
    #             lr = cv2.resize(hr, dsize=[64,64], interpolation=cv2.INTER_LINEAR)
    #             lr_dir = os.path.join(lrcls_dir, hr_nm)
    #             cv2.imwrite(lr_dir, lr)
    #             img_paths_lq.append(lr_dir)
    #DIOR
    gt_dirs = sorted(glob.glob(os.path.join(args.input_gt, '*.jpg')))
    os.makedirs(args.input_lr, exist_ok=True)
    for i,gt_dir in enumerate(gt_dirs):
        gt = cv2.imread(gt_dir)
        gt_nm = os.path.basename(gt_dir)
        lr_dir = os.path.join(args.input_lr, gt_nm)
        lr = cv2.resize(gt, dsize=[200,200], interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(lr_dir, lr)
        img_paths_lq.append(lr_dir)
        img_paths_gt.append(gt_dir)

# make sure put the paths of LR images before HR images, related to loading dataset
    for img_path_gt, img_path_lq in zip(img_paths_gt, img_paths_lq):
        print(f'{img_path_lq}, {img_path_gt}')
        txt_file.write(f'{img_path_lq},{img_path_gt}\n')






if __name__ == '__main__':
    """This script is used to generate meta info (txt file) for paired images.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_gt',
        nargs='+',
        default='')
    parser.add_argument(
        '--input_lr',
        nargs='+',
        default='')
    parser.add_argument('--root', nargs='+', default=[None, None], help='Folder root, will use the ')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='',
        help='txt path for meta info')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    main(args)
