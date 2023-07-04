#ifndef _THREE_NN_WRAPPER_FAST_KERNEL
#define _THREE_NN_WRAPPER_FAST_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
  at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown,
	const float *known, float *dist2, int *idx, cudaStream_t stream);
    
#ifdef __cplusplus
}
#endif

#endif