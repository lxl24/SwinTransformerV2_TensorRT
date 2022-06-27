import tensorrt as trt
import numpy as np
from collections import OrderedDict
import onnx
import onnx_graphsurgeon as gs


onnxFile='/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swin_12.onnx'
graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(onnxFile)))

LN_input_node=[]
LN_output_node=[]

for node in graph.nodes:
    if node.op == 'Gather': 
    #   print(node.o(0).name)
      if node.o(0).op == 'ReduceL2' and node.o(1).op == 'Shape' and node.o(2).op == 'Div':

    
        LN_input_node.append(node)
        
        lastDivNode = node.o(2)
        # print(lastDivNode.name)
        LN_output_node.append(lastDivNode)

    #     lastDivNode.outputs = []
    # if node.op == 'Add' and node.o().op == "ReduceMean":
    #     LN_input_node.append(node)
    # if node.op == 'Div' and node.o().op == "Mul":
    #     LN_output_node.append(node.o())
        
# print(num_of_LN)       
       
num_of_LN=len(LN_input_node)   
     
for i in range(num_of_LN):
    # print(LN_input_node[i].name,LN_output_node[i].name)
    LN_out = gs.Variable(name="Fnorm_out"+str(i), dtype=np.dtype(np.float32), shape=None)
    LN_node = gs.Node(name="Fnorm"+str(i), op="L2Norm", inputs=[LN_input_node[i].outputs[0]], outputs=[LN_output_node[i].outputs[0]])
    graph.nodes.append(LN_node)
    LN_output_node[i].outputs=[]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "models/checkpoints/swin_12_FNorm.onnx")
print("Succeeded inserting fnorm node!")