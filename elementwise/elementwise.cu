/*
 * @Author: Zhou Zijian 
 * @Date: 2025-02-23 00:42:46 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2025-02-23 00:43:06
 */

#include <elementwise.h>

// 元素加法的CUDA核函数
__global__ void ElementwiseAddKernel(float *a, float *b, float *c, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) {
        return;
    }
    c[idx] = a[idx] + b[idx];
}

// 定义FLOAT4宏，将float指针转换为float4指针
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// 4元素加法的CUDA核函数
__global__ void ElementwiseAddx4Kernel(float *a, float *b, float *c, int size)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (idx >= size) {
        return;
    }
    float4 a4 = FLOAT4(a[idx]);
    float4 b4 = FLOAT4(b[idx]);
    float4 c4;
    c4.x = a4.x + b4.x;
    c4.y = a4.y + b4.y;
    c4.z = a4.z + b4.z;
    c4.w = a4.w + b4.w;
    FLOAT4(c[idx]) = c4;
}

// 元素加法的主机函数
void ElementwiseAdd(float *a, float *b, float *c, int size)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    ElementwiseAddKernel<<<gridSize, blockSize>>>(a, b, c, size);
    cudaDeviceSynchronize();
}

void ElementwiseAddAsync(float *a, float *b, float *c, int size, cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    ElementwiseAddKernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, size);
}

// 4元素加法的主机函数
void ElementwiseAddx4(float *a, float *b, float *c, int size)
{
    constexpr int blockSize = 256 / 4;
    int gridSize = (size + blockSize - 1) / blockSize;
    ElementwiseAddx4Kernel<<<gridSize, blockSize>>>(a, b, c, size);
    cudaDeviceSynchronize();
}

void ElementwiseAddx4Async(float *a, float *b, float *c, int size, cudaStream_t stream)
{
    constexpr int blockSize = 256 / 4;
    int gridSize = (size + blockSize - 1) / blockSize;
    ElementwiseAddx4Kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, size);
}