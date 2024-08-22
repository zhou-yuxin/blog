#include <stdio.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_fp16.h>

// for the parameter structure
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"

// a short name to the parameter structure
using fmha_param_t = tensorrt_llm::kernels::Fused_multihead_attention_params_v2;

// declare the cubin in another file
namespace tensorrt_llm {
namespace kernels {
    extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89_cu_cubin[];
}};
// a short name to the cubin
const void* cubin = tensorrt_llm::kernels::cubin_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89_cu_cubin;
// the function (aka kernel) name in the cubin
const char* func_name = "fmha_v2_flash_attention_fp16_64_64_S_qkv_16_causal_sm89_kernel_nl";

int main() {
    // Init the CUDA env
    CUresult result = cuInit(0);
    assert(result == CUDA_SUCCESS);

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    assert(result == CUDA_SUCCESS);

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    assert(result == CUDA_SUCCESS);

    // load cubin module from memory
    CUmodule module;
    result = cuModuleLoadData(&module, cubin);
    assert(result == CUDA_SUCCESS);

    // load the desired function from this module by name
    CUfunction function;
    result = cuModuleGetFunction(&function, module, func_name);
    assert(result == CUDA_SUCCESS);

    // prepare a packed QKV matrix.
    // each of Q,K,V has shape of [seq=64, dim=16], the packed matrix has [64, 48]
    const size_t s = 64, d = 16;
    const size_t qkv_size = sizeof(half) * s * d * 3;
    half* qkv_host = (half*)malloc(qkv_size);
    assert(qkv_host != nullptr);
    // random fill, {0.00, 0.01, 0.02, ... 0.99}
    srand(1234);
    for(size_t i = 0; i < s * d * 3; i++) {
        qkv_host[i] = (rand() % 100) * 0.01f;
    }
    // dump them, so we can valid the result in python
    auto dump_matrix = [=](const char* name, const half* matrix) {
        printf("dump %s:\n", name);
        const size_t stride = 3 * d;
        for(size_t row = 0; row < s; row++) {
            for(size_t col = 0; col < d; col++) {
                printf("%.2f ", float(matrix[row * stride + col]));
            }
            printf("\n");
        }
    };
    dump_matrix("Q", qkv_host);
    dump_matrix("K", qkv_host + d);
    dump_matrix("V", qkv_host + 2 * d);

    // copy packed QKV to device
    void* qkv_device;
    cudaError_t error = cudaMalloc(&qkv_device, qkv_size);
    assert(error == 0);
    error = cudaMemcpy(qkv_device, qkv_host, qkv_size, cudaMemcpyHostToDevice);
    assert(error == 0);

    // the output matrix O = softmax(QK * scale) * V, shape [seq, dim]
    void* o_device;
    const size_t o_size = sizeof(half) * s * d;
    error = cudaMalloc(&o_device, o_size);
    assert(error == 0);

    // fill the parameter structure
    fmha_param_t params;
    params.b = 1;
    params.h = 1;
    params.s = s;
    params.d = d;
    params.qkv_ptr = qkv_device;
    params.qkv_stride_in_bytes = sizeof(half) * d * 3;
    params.o_ptr = o_device;
    params.o_stride_in_bytes = sizeof(half) * d;
    // ***
    // balabala

#if 0
    void* kernel_params[] = {&params, nullptr};
    cuLaunchKernel(function,
        1, 1, 1,    // 1 block is needed
        128, 1, 1,  // 128 threads
        1234,       // shared memory size
        nullptr,    // CUDA stream
        kernel_params,
        nullptr);
#endif

    return 0;
}