# SwinTransformerV2_TensorRT

## SwinTransformerV2
### 模型简介

- The Swin Transformer is a type of Vision Transformer. It builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red). It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks. In contrast, previous vision Transformers produce feature maps of a single low resolution and have quadratic computation complexity to input image size due to computation of self-attention globally.
  
![avatar](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png)



### 模型优化的难点
