# cubin的使用——以fmha为例

上一篇《[动态链接的本质](./动态链接的本质/README.md)》讲了动态库的原理，现在可以介绍一下cubin如何使用了。其实cubin与Linux上常用的so非常相似，有多像呢？
* so是ELF(Executable and Linkable Format)格式的文件，包含了CPU能够执行的指令；</br>cubin也是ELF文件，包含了GPU能够执行的指令。
* 使用`gcc`编译`.c`文件，默认将函数嵌在可执行文件中，除非加了`-shared`参数，就会生成可共享的so文件；</br>使用`nvcc`编译`.cu`文件，默认将kernel嵌在可执行文件中，除非加了`-cubin`参数，就会生成单独的cubin文件。
* 可以使用`objdump`命令查看so文件包含的各个段与导出的函数，甚至反汇编之；</br>可以使用`cuobjdump`命令查看cubin文件包含的哥哥段与导出的kernel，甚至反汇编之。

不说一模一样吧，也只能说毫无区别了:-)，cubin就是CUDA体系的动态链接库。如果你知道如何动态加载so文件中的函数，那么加载cubin中的kernel也是易如反掌。

哪里有高质量的cubin让我们小试牛刀一下呢？把目光看向[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)自带的fmha（fused multihead attention）呀，里面是multihead attention的高效实现。

进入[cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin)目录，里面是琳琅满目的fmha实现，总有一款适合你。其命名规则为（只考虑较新的version 2, flash attention实现）：
```
fmha_v2_flash_attention_<dtype>_<q_step>_<kv_step>_S_<qkv_layout>_<dim>_[attr]_<sm>.cubin.cpp
```
* dtype包括`bf16`、`fp16`、`e4m3`等，如果dtype以`_fp32`结尾，意味着计算softmax后保留的格式是fp32，否则为fp16。通常fp32精度会更加，代价是显存占用也会更多。
* q_step是指Q按行切分粒度。假设seq_length=1024，而q_step=64，则计算过程会按行切成1024/64=16份并行计算。
* kv_step是指KV按列切分粒度。假设seq_length=1024，而kv_step=32，则计算过程会按列迭代1024/32=32次，或者切成32份并行计算（是单个CTA迭代还是多个CTA并行计算最后整合，取决于一些参数）。
* dim是QKV的维度，比如llama模型采用的是128。
* sm是目标GPU架构，比如Ampere是sm80，Ada是sm89，Hopper是sm90。
* qkv_layout是QKV矩阵的排布方式，`qkv`意味着QKV是按列拼接的连续存储（packed QKV），`q_kv`是指Q是独立的，而KV是按列拼接的连续存储（Q + contiguous KV），`q_paged_kv`是指Q是独立的，KV采用paged KV的形式，即有一个指针数组，每个指针指向一个块，每个块由固定数量的K或V构成。由于指针数组像OS的页表，而KV块像OS中的页，因此得名。

我们就选一个简单些、维度低些的kernel：[fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89.cubin.cpp](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89.cubin.cpp)试验一下吧。
```
# make and enter a new workspace
mkdir test
cd test
# link it here for easy access
ln -s path/to/fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89.cubin.cpp .
```
之后编写`test.cpp`如下：
```cpp
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
    cudaError_t error = cudaMalloc(&o_device, o_size);
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
    // balabala 未完待续

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
```

之后编译
```
nvcc test.cpp fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89.cubin.cpp \
    -Ipath/to/trtllm/cpp -Ipath/to/trtllm/cpp/include
    -o test -lcuda
```