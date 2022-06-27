import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from glob import glob
from time import time_ns,perf_counter
import torch



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


mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
batch_size = [1,4,8,16]
img_path_list = glob("/root/workplace/imagenet/train/test/*.JPEG")


onnx_model = onnx.load_model("/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12.onnx")
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









