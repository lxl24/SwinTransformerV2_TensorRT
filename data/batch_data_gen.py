import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from glob import glob
from time import time_ns,perf_counter
from utils import imread, normalize
import argparse
import torch



def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer export script', add_help=False)
    parser.add_argument('--model', type=str,default="/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12.onnx", metavar="FILE", help='path to config file', )
    args, unparsed = parser.parse_known_args()
    return args

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
batch_size = [1,4,8,16]                # 生成四种batch的数据


args = parse_option()

img_path_list = glob("/root/workplace/imagenet/train/test/*.JPEG")
onnx_model = onnx.load_model(args.model)
options = ort.SessionOptions()
sess = ort.InferenceSession(
    onnx_model.SerializeToString(), options,providers=[('CPUExecutionProvider')
])

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

for batch in batch_size:
    img_list = []
    for im in img_path_list[: batch]:
        img = imread(im)
        img_list.append(img)
    data= np.concatenate(img_list, axis=0)
    print("data_shape:",data.shape)
    output = sess.run([output_name], {input_name : data})
    output = np.squeeze(output[0])
    np.savez("data/swin-b{}".format(batch), input=data, output=output)









