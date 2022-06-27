import torch 
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from time import time_ns,perf_counter
from models.swin_v2 import SwinTransformerV2
from config import get_config
from models import build_model
from utils import load_checkpoint
import torch
import argparse
 
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
    parser.add_argument('--type', default='swinv1', type=str, help='path to dataset')

    # settings for exporting onnx
    parser.add_argument('--batch-size-onnx',default=32, type=int, help="batchsize when export the onnx model")


    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def pytorch_inference(config,args):
    if args.type == "swinv1":
        dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    if args.type == "swinv2":
        dummy_input = torch.randn(1, 3, 256, 256, device='cuda')
    model = build_model(config)
    model.eval()
    model.to('cuda')
    max_accuracy = load_checkpoint(config, model, None, None)

    for i in range(30):
        with torch.no_grad():
            output = model(dummy_input)

    torch.cuda.synchronize()    
    st_time =  perf_counter()
    for i in range(1000):
        with torch.no_grad():
            output = model(dummy_input)
    torch.cuda.synchronize()
    ed_time = perf_counter()
    print("This session's running time:",(ed_time - st_time)/1000)

     # 清空释放显存
    del model
    dummy_input.cpu()
    del dummy_input
    torch.cuda.empty_cache()



if __name__ == '__main__':
    args, config = parse_option()

    pytorch_inference(config,args)