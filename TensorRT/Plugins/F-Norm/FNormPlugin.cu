/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "FNormPlugin.h"
#include <cub/cub.cuh>



using namespace nvinfer1;


PluginFieldCollection L2NormPluginCreator::fc_{};
std::vector<PluginField> L2NormPluginCreator::attr_;

template<typename T>
__device__ T GetZeroVal() {
  return static_cast<T>(0);
}


template<typename T, int TPB>
__global__ void L2Normalize(const int32_t n, const int32_t c, const int32_t d,
                                   const T epsilon, const T* in, T* square_x_sum, T* out) {
  using BlockReduce = cub::BlockReduce<T, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    // sum is 0
    T sum = GetZeroVal<T>();

    // offset is ?
    const int32_t offset = (i / d) * d * c + (i % d);
    
    // j is tidx 
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const T x = in[offset + j * d];
      sum += x * x;
    }

    const T reduce_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) { square_x_sum[i] = reduce_sum; }
    __syncthreads();

    const T inv_norm = rsqrtf(fmaxf(square_x_sum[i], epsilon));
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const int32_t index = offset + j * d;
      out[index] = inv_norm * in[index];
    }
  }
}


// template<typename T>
// static void L2NormalizeForward(const int32_t n, const int32_t c, const int32_t d, const T epsilon,
//                                const T* in, T* square_x_sum, T* out) {
//   for (int32_t i = 0; i < n; i++) {
//     const int32_t offset = (i / d) * d * c + (i % d);
//     for (int32_t j = 0; j < c; j++) {
//       const T x = in[offset + j * d];
//       square_x_sum[i] += x * x;
//     }
//     const T norm = std::sqrt(std::max(square_x_sum[i], epsilon));
//     for (int32_t j = 0; j < c; j++) {
//       const int32_t index = offset + j * d;
//       out[index] = in[index] / norm;
//     }
//   }
// }

template __global__ void L2Normalize<float,64>(const int32_t , const int32_t , const int32_t ,const float , const float* , float* , float* );
template __global__ void L2Normalize<half,32>(const int32_t , const int32_t , const int32_t ,const half , const half* , half* , half* );


int32_t L2NormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    const int32_t  n = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    const int32_t  c = inputDesc[0].dims.d[2];
    const int32_t  d = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] *  inputDesc[0].dims.d[2];
    // auto*    square_x_sum   = reinterpret_cast<float *>(workspace);
    float * square_x_sum;
    cudaMalloc((void **)&square_x_sum, sizeof(float));
    auto  epsilon =6e-12;

    constexpr int VPT = 16;
    constexpr int TPB = 256;
    (L2Normalize<float, TPB>)<<<gridSize, TPB, 0, stream>>>  ((const int32_t)n ,(const int32_t)c, (const int32_t)d, (const float)epsilon,(const float*)inputs[0],(float*)square_x_sum , (float*)outputs[0]);

    // if (inputDesc[0].type == DataType::kFLOAT)
    // {
    //     constexpr int VPT = 16;
    //     constexpr int TPB = 256;
    //     (L2Normalize<float, TPB>)<<<gridSize, TPB, 0, stream>>>  ((const int32_t)n ,(const int32_t)c, (const int32_t)d, (const float)epsilon,(const float*)inputs[0],(float*)square_x_sum , (float*)outputs[0]);
    // }
    // else
    // {
    //     constexpr int VPT = 16;
    //     constexpr int TPB = 256 ;
    //     (L2Normalize<half,TPB>)<<<gridSize, TPB, 0, stream>>>  ((const int32_t)n ,(const int32_t)c, (const int32_t)d, (const half)epsilon,(const half*)inputs[0],(half*)square_x_sum , (half*)outputs[0]);
    // }

    cudaFree(square_x_sum);
    return 0;
}



REGISTER_TENSORRT_PLUGIN(L2NormPluginCreator);


