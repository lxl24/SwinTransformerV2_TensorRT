trtexec  --onnx=/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swinv1_layerNorm12.onnx \
      --fp16  \
      --minShapes=input:1x3x224x224  \
      --optShapes=input:8x3x224x224  \
      --maxShapes=input:16x3x224x224  \
      --calib=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/INT8/swin_calibration_test.cache \
      --workspace=230000000000 \
      --saveEngine=TensorRT/TRT_Engine/swinv1_12_fp16.plan \
      --plugins=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/Plugins/LayerNormPlugin_with_params/LayerNorm.so \
      --verbose    \
      --dumpProfile \
      > Results/trt_log/swinv1_trt_report.txt 