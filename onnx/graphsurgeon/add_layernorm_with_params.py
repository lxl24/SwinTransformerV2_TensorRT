import tensorrt as trt
import numpy as np
from collections import OrderedDict
import onnx
import onnx_graphsurgeon as gs
from collections import OrderedDict
from copy import deepcopy

onnxFile='/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12.onnx'
graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(onnxFile)))

LN_input_node=[]
LN_output_node=[]

nLayerNormPlugin = 0

for node in graph.nodes:
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
            node.o().o(0).o().o().o().o().o().op == 'Mul' and \
            node.o().o(0).o().o().o().o().o().o().op == 'Add':

            inputTensor = node.inputs[0]

            lastMultipyNode = node.o().o(0).o().o().o().o().o()
            index = ['weight' in i.name for i in lastMultipyNode.inputs].index(True)
            b = np.array(deepcopy(lastMultipyNode.inputs[index].values.tolist()), dtype=np.float32)
            constantB = gs.Constant("LayerNormB-" + str(nLayerNormPlugin), np.ascontiguousarray(b.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

            lastAddNode = node.o().o(0).o().o().o().o().o().o()
            index = ['bias' in i.name for i in lastAddNode.inputs].index(True)
            a = np.array(deepcopy(lastAddNode.inputs[index].values.tolist()), dtype=np.float32)
            constantA = gs.Constant("LayerNormA-" + str(nLayerNormPlugin), np.ascontiguousarray(a.reshape(-1)))

            inputList = [inputTensor, constantB, constantA]
            layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), None)
            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, outputs=[layerNormV])
            graph.nodes.append(layerNormN)
            nLayerNormPlugin += 1

            # print(layerNormN.op)

            for n in graph.nodes:
                if lastAddNode.outputs[0] in n.inputs:
                    index = n.inputs.index(lastAddNode.outputs[0])
                    n.inputs[index] = layerNormN.outputs[0]
            lastAddNode.outputs = []
            continue
        

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "models/checkpoints/swinv1_layerNorm12.onnx")
print("Succeeded inserting layernorm node!")