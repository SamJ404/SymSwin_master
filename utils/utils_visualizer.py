import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as transform
import timm
from collections import OrderedDict
from PIL import Image

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image

from models.network_swinir import SwinIR as net
# from models.swin_hw_csffa import Swin_hw_csffa as net

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image_path',
        type=str,
        default='/home/sam/SuperRes/srtest/vis/harbor_117x4.png',
        help='Input image path')
    parser.add_argument(
        '--imagename',
        type=str,
        default='harbor_117x4',
        help='Input image path')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=64, width=64):
   ##第一层height、wideth设置为28，第二层height、wideth设置为14，第三、四层height、wideth设置为7
    # result = tensor.reshape(tensor.size(0),
    #                         height, width, tensor.size(1))
    #
    # # Bring the channels to the first dimension,
    # # like in CNNs.
    # result = result.transpose(2, 3).transpose(1, 2)
    result = tensor
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    trans = transform.Compose([transform.ToTensor(), ])

    model = net()
    ckpt = torch.load('/home/sam/SuperRes/realesrgan/test_models/swinir/500000_G_df2k.pth',map_location='cpu')
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    target_layers = [[model.layers[0].conv], [model.layers[1].conv], [model.layers[2].conv], [model.layers[3].conv], [model.layers[4].conv], [model.layers[5].conv]]


    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    for i, target_layer in enumerate(target_layers):
        cam = methods[args.method](model=model,
                                   target_layers=target_layer,
                                   reshape_transform=reshape_transform)

        lr = Image.open(args.image_path).convert('RGB')
        lr = trans(lr).unsqueeze(0).to('cuda:0')
        pred = model(lr)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        target_category = None

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1

        grayscale_cam = cam(input_tensor=lr,
                            targets=target_category,
                            eigen_smooth=False,
                            aug_smooth=False)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(lr, grayscale_cam)
        cv2.imwrite(f'{args.method}_{args.imagename}_swinirlayer{i}.jpg', cam_image)

