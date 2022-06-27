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
    
    # test infernece time
    for i in range(10):
            output = sess.run([output_name], {input_name : img})
        
    st_time =  perf_counter()
    for i in range(30)
        output = sess.run([output_name], {input_name : img})
        torch.cuda.synchronize()
        ed_time = perf_counter()
        print("This session's running time:",(ed_time - st_time))
        output = np.squeeze(output[0])
    return output