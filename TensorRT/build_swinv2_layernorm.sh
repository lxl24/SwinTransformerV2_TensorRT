trtexec  --onnx=/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swin_12_layerNorm.onnx \
      --minShapes=input:1x3x256x256  \
      --optShapes=input:8x3x256x256  \
      --maxShapes=input:16x3x256x256  \
      --calib=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/INT8/swin_calibration_test.cache \
      --workspace=23000000000 \
      --saveEngine=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/TRT_Engine/swin_12_layerNorm_wp.plan \
      --plugins=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/Plugins/LayerNormPlugin_with_params/LayerNorm.so \
      --verbose    \
      --dumpProfile \
      > Results/trt_log/swinv1_trt_report.txt 