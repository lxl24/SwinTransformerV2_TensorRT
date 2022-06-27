# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import argparse
import torch
import onnx
from config import get_config
from models import build_model
from utils import load_checkpoint, imread,normalize



def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer export script', add_help=False)
    parser.add_argument('--cfg', type=str,default="/root/workplace/SwinTransformerV2_TensorRT/models/swin.yaml", metavar="FILE", help='path to config file', )
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
    parser.add_argument('--resume', default='/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swin_small_patch4_window7_224.pth', help='resume from checkpoint')
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
    parser.add_argument('--type', default='swinv1', type=str, help='path to dataset')
    

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def export_onnx(model, config,args):
    # ONNX export
    try:
        model=model.eval()
        if args.type == "swinv1":
            f = "models/checkpoints/swinv1_12.onnx"
            dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
        if args.type == "swinv2":
            f = "models/checkpoints/swinv2_12.onnx"
            dummy_input = torch.randn(1, 3, 256, 256, device='cpu')

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        # f = config.MODEL.RESUME.replace('.pth', '.onnx')  # filename
        # f = "models/checkpoints/swinv1_12.onnx"
        input_names = ["input"]
        output_names = ["output"]

     
        dynamic_axes = {'input': {0: 'batch_size'},'output': {0: 'batch_size'}}
        torch.onnx.export(model, dummy_input, f, verbose=False, opset_version=12,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                        #   enable_onnx_checker=False,
                          do_constant_folding=True,
                        #   export_params=True,
                        #   keep_initializers_as_inputs=True,
                          )
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)


def main(config,args):

    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model = model.eval()
    model.to('cpu')
    max_accuracy = load_checkpoint(config, model, None, None)
    export_onnx(model, config,args)


if __name__ == '__main__':
    args, config = parse_option()
    main(config,args)