CUDA_PATH       = /usr/local/cuda
TRT_PATH        = /usr/local/TensorRT-8.4.1.4/
NVCC            = $(CUDA_PATH)/bin/nvcc
#SM              = 61
                # 61 for GTX1070, 75 for T4,80 for A30
GENCODE         = -gencode=arch=compute_86,code=sm_86
CUFLAG          = -w -std=c++14 -O3 -UDEBUG -Xcompiler -fPIC $(GENCODE)
CPPFLAG         = -w -std=c++14 -O3 -use_fast_math
SOFLAG          = $(CUFLAG) -shared
INCLUDE         = -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -L$(TRT_PATH)/lib -lnvinfer

SRC_CU          = $(shell find ./ -name '*.cu')

all: LayerNorm.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

LayerNorm.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt

.PHONY: test
test:
	clear
	python testLayerNormPlugin.py

