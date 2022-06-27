import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from time import time_ns,perf_counter
from models.swin_v2 import SwinTransformerV2
from config import get_config
from models import build_model
from utils import load_checkpoint
from export_onnx import parse_option
import torch

## macro define
mean = np.array([0.5, 0.5, 0.5])
std  = np.array([0.5, 0.5, 0.5])
onnx_path = "/root/workplace/SwinTransformerV2_TensorRT/models/checkpints/swinv2_small_patch4_window8_256.onnx"
pth_path = "/root/workplace/SwinTransformerV2_TensorRT/models/checkpints/swinv2_small_patch4_window8_256.pth"
img_path = "/root/workplace/imagenet/train/test/ILSVRC2012_test_00000001.JPEG"

def imread(img_path):
    img = np.array(Image.open(img_path).convert("RGB").resize((224,224)))
    img = normalize(img, mean, std)
    img = np.expand_dims(img, 0)
    return img.astype(np.float32)

def normalize(data, mean, std):
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(std, np.ndarray):
        std = np.array(std)
    if mean.ndim == 1:
        mean = np.reshape(mean, (1, 1, -1))
    if std.ndim == 1:
        std = np.reshape(std, (1, 1, -1))
    norm = np.divide(np.subtract(np.divide(data, np.max(abs(data))),mean),std).transpose((2, 0, 1))  
    return norm 

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

def onnx_inference(img):

    providers = [
	  ('CUDAExecutionProvider', {
		'device_id': 0,
	  })
    ]
    onnx_model = onnx.load_model("/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12.onnx")
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
    _, config = parse_option()
    img = imread(img_path)
    output1 = pytorch_inference(img,config)
    print(np.max(output1))
    output2 = onnx_inference(img)
    print(np.max(output2))
    res, diff0, diff1 = check(output1,output2)
    print(res, diff0, diff1)
    if not np.allclose(output1, output2,rtol=1.e-5,atol=1e-05):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
    print('The outputs are same between Pytorch and ONNX')