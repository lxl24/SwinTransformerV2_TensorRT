trtexec  --onnx=/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swin_12.onnx \
      --minShapes=input:1x3x224x224  \
      --optShapes=input:8x3x224x224  \
      --maxShapes=input:16x3x224x224  \
      --calib=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/INT8/swin_calibration_test.cache \
      --workspace=23000000000 \
      --saveEngine=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/TRT_Engine/swinv1_12.plan \
      --verbose    \
      --dumpProfile \
      > Results/trt_log/swinv1_trt_report.txt 