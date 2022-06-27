#!/usr/bin/python
import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt

dataFilePath = "/root/workplace/SwinTransformerV2_TensorRT/data/"
planFilePath   = "/root/workplace/SwinTransformerV2_TensorRT/TensorRT/TRT_Engine/"
pluginPath = "/root/workplace/SwinTransformerV2_TensorRT/TensorRT/Plugins/LayerNormPlugin_with_params/"
ResultPath = "/root/workplace/SwinTransformerV2_TensorRT/Results/"

PlanFile  = planFilePath + "swinv1_12_layernorm_fp16.plan"
ResultFile = ResultPath  + "encoderScore.txt"
soFileList = glob(pluginPath + "*.so")

tableHead = \
"""
bs: Batch Size
lt: Latency (ms)
tp: throughput (img/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+--------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0| output check
----+--------+---------+---------+---------+-------------
"""

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])


def check(a, b, weak=True, epsilon = 1e-5):

    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )

    diff0 = np.max(np.abs(a - b))

    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    return res, diff0, diff1

#-------------------------------------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

#-------------------------------------------------------------------------------

print("Test Part!")

with open(ResultFile, 'w') as f:

    if os.path.isfile(PlanFile):
        with open(PlanFile, 'rb') as swin:
            engine = trt.Runtime(logger).deserialize_cuda_engine(swin.read())
        if engine is None:
            print("Failed loading %s"%PlanFile)
            exit()
        print("Succeeded loading %s"%PlanFile)
    else:
        print("Failed finding %s"%PlanFile)
        exit()

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    context = engine.create_execution_context()
        
    print(tableHead)  # for standard output

    for ioFile in sorted(glob(dataFilePath + "./swin-*.npz")):
        ioData = np.load(ioFile)
        # print(ioData.files)
        input = ioData['input']

        batchSize, _, _, _ = input.shape
        if batchSize > 16:
            continue

        context.set_binding_shape(0, input.shape)

        
        bufferH = []
        bufferH.append(input.astype(np.float32).reshape(-1) )

        for i in range(nInput, nInput + nOutput):                
            bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

        bufferD = []
        for i in range(nInput + nOutput):                
            bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # warm up
        for i in range(10):
            context.execute_v2(bufferD)

        # test infernece time
        t0 = time_ns()
        for i in range(30):
            context.execute_v2(bufferD)
        t1 = time_ns()
        timePerInference = (t1-t0)/1000/1000/30

        indexEncoderOut = engine.get_binding_index('output')
        trtout = np.squeeze(bufferH[indexEncoderOut])
        # print(trtout)
        # print(ioData['output'])
        check0 = check(trtout,ioData['output'],True,5e-5)

        string = "%4d,%8.3f,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                    timePerInference,
                                                    batchSize/timePerInference*1000,
                                                    check0[1],
                                                    check0[2])
        
        print(string + ", %s"%("Good" if check0[1] < 1e-5 and check0[2] < 2e-3 else "Bad"))
        f.write(string + "\n")

        for i in range(nInput + nOutput):                
            cudart.cudaFree(bufferD[i])
