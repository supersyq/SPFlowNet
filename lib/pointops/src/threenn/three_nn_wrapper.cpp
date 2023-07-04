#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "three_nn_wrapper_kernel.h"

extern THCState *state;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {

    CHECK_INPUT(unknown_tensor);
    CHECK_INPUT(known_tensor);

    const float *unknown = unknown_tensor.data<float>();
    const float *known = known_tensor.data<float>();
    float *dist2 = dist2_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);

    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx, stream);
}