import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from time import time_ns,perf_counter
from models.swin_v2 import SwinTransformerV2
from config import get_config
from models import build_model
from utils import load_checkpoint,imread,normalize
from export_onnx import parse_option
import torch
import argparse

## macro define
mean = np.array([0.5, 0.5, 0.5])
std  = np.array([0.5, 0.5, 0.5])
img_path = "/root/workplace/imagenet/train/test/ILSVRC2012_test_00000001.JPEG"


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
    parser.add_argument('--onnx', default='/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12.onnx', help='resume from checkpoint')
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


def pytorch_inference(img,config):
    seed = 0  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


    img = torch.from_numpy(img)
    print(img)
    model = build_model(config)
    model = model.eval()
    model.to('cpu')
    max_accuracy = load_checkpoint(config, model, None, None)
    with torch.no_grad():
        output = model(img)
    output = np.squeeze(output.numpy())
    return output

def onnx_inference(img,models):

    providers = [
	  ('CUDAExecutionProvider', {
		'device_id': 0,
	  })
    ]
    onnx_model = onnx.load_model(models)
    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # st_time =  perf_counter()
    output = sess.run([output_name], {input_name : img})
    # torch.cuda.synchronize()
    # ed_time = perf_counter()
    # print("This session's running time:",(ed_time - st_time))
    output = np.squeeze(output[0])
    return output

def check(a, b, weak=True, epsilon = 1e-5):

    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )

    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    return res, diff0, diff1


if __name__ == '__main__':
    args, config = parse_option()
    img = imread(img_path)
    output1 = pytorch_inference(img,config)
    output2 = onnx_inference(img,args.onnx)  
    res, diff0, diff1 = check(output1,output2)
    print(res, diff0, diff1)
    if not np.allclose(output1, output2,rtol=1.e-5,atol=1e-05):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
    print('The outputs are same between Pytorch and ONNX')