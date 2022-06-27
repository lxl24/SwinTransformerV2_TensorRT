import numpy as np
import onnx
import onnx_graphsurgeon as gs


graph = gs.import_onnx(onnx.load("/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12.onnx"))
nLayerNormPlugin = 0

for node in graph.nodes:
        # replace layernorm
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):

            inputTensor = node.inputs[0]
            print(node.name)
            lastDivNode = node.o().o(0).o().o().o().o()

            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=[inputTensor], outputs=[lastDivNode.outputs[0]])
            graph.nodes.append(layerNormN)
            nLayerNormPlugin += 1

            lastDivNode.outputs = []

            continue   
             
print(nLayerNormPlugin)
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_12_layerNorm.onnx")
print("pass")