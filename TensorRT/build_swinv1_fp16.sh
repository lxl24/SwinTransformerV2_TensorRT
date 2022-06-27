trtexec  --onnx=/root/workplace/SwinTransformerV2_TensorRT/models/checkpoints/swin_12.onnx \
      --fp16  \
      --minShapes=input:1x3x224x224  \
      --optShapes=input:1x3x224x224  \
      --maxShapes=input:4x3x224x224  \
      --calib=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/INT8/swin_calibration_test.cache \
      --workspace=23000000000 \
      --saveEngine=/root/workplace/SwinTransformerV2_TensorRT/TensorRT/TRT_Engine/swinv1_12_fp16.plan \
      --verbose    \
      --dumpProfile \
      > Results/trt_log/swinv1_trt_report.txt 