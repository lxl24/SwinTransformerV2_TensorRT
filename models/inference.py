import torch
from models.swin_v2 import SwinTransformerV2
import os
import argparse
from config import get_config
from models import build_model
from utils import load_checkpoint
import PIL.Image as Image
import numpy as np
import cv2
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer export script', add_help=False)
    parser.add_argument('--cfg',default="/root/workplace/SwinTransformerV2_TensorRT/models/swinv2.yaml",type=str, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='../imagenet_1k', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', default='/root/workplace/SwinTransformerV2_TensorRT/checkpints/swinv2_small_patch4_window8_256.pth', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # settings for exporting onnx
    parser.add_argument('--batch-size-onnx',default=32, type=int, help="batchsize when export the onnx model")


    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return config


def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


def build_transform():
    t = []
    DATA_IMG_SIZE=256
    # size = int((224/ 208) * DATA_IMG_SIZE)
    t.append(
        transforms.Resize((DATA_IMG_SIZE,DATA_IMG_SIZE), interpolation=_pil_interp("bicubic")),
        # to maintain same ratio w.r.t. 224 images
    )
    # t.append(transforms.RandomCrop(DATA_IMG_SIZE))
    # t.append(transforms.RandomVerticalFlip(p=0.5))
    # t.append(transforms.RandomHorizontalFlip(p=0.5))
    # t.append(transforms.RandomRotation(degrees=(10, 80)))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(0.5, 0.5))
    return transforms.Compose(t)    

if __name__ == "__main__":
    config = parse_option()
    model = build_model(config)
    max_accuracy = load_checkpoint(config, model, None, None)
    
    img = Image.open("/root/workplace/imagenet/train/test/ILSVRC2012_test_00000001.JPEG").convert("RGB")
    trans = build_transform()
    img = trans(img)
    img = torch.unsqueeze(img,0)
    print(img.shape)
    out = model(img)
    print(out.shape)



