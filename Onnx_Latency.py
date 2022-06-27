import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from time import time_ns,perf_counter
from models.swin_v2 import SwinTransformerV2
from config import get_config
from models import build_model
from utils import imread, normalize
import torch


img_path = "/root/workplace/imagenet/train/test/ILSVRC2012_test_00000001.JPEG"
mean = np.array([0.5, 0.5, 0.5])
std  = np.array([0.5, 0.5, 0.5])

def onnx_inference(img):

    providers = [
	  ('CUDAExecutionProvider', {
		'device_id': 0,
	  })
    ]
    onnx_model = onnx.load_model("/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12.onnx")
    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=providers)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # test infernece time
    for i in range(10):
            output = sess.run([output_name], {input_name : img})
    torch.cuda.synchronize()    
    st_time =  perf_counter()
    for i in range(30):
        output = sess.run([output_name], {input_name : img})
    torch.cuda.synchronize()
    ed_time = perf_counter()
    print("This session's running time:",(ed_time - st_time)/30)
  
   


if __name__ == '__main__':
    img = imread(img_path)
    onnx_inference(img)
