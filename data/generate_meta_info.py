# by SAM J

import argparse
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')
    for folder in args.input:
        # dior
        img_paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        for img_path in img_paths:
            print(img_path)
            txt_file.write(f'{img_path}\n')
        # resisc
        # for cls in sorted(os.listdir(folder)):
        #     cls_dir = os.path.join(folder, cls)
        #     img_paths = sorted(glob.glob(os.path.join(cls_dir, '*.png')))
        #     for i, img_path in enumerate(img_paths):
        #         if i == 364 or i == 634 or i%100 == 0:
        #             continue
        #         else:
        #             print(img_path)
        #             txt_file.write(f'{img_path}\n')


if __name__ == '__main__':
    """Generate meta info (txt file) for only Ground-Truth images.

    It can also generate meta info from several folders into one txt file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=[''],
        help='Input folder, can be a list')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()


    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    main(args)
