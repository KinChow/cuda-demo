/*
 * @Author: Zhou Zijian 
 * @Date: 2025-02-23 00:43:13 
 * @Last Modified by:   Zhou Zijian 
 * @Last Modified time: 2025-02-23 00:43:13 
 */

#include <cassert>
#include <iostream>
#include "elementwise.h"

// 检查CUDA错误的宏定义
#define CUDA_CHECK_ERROR(err)                                                                      \
    if (err != cudaSuccess) {                                                                      \
        std::cerr << "Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1);                                                                                   \
    }

int main()
{
    cudaStream_t stream;
    cudaError_t cudaStatus;
    cudaStatus = cudaStreamCreate(&stream);
    CUDA_CHECK_ERROR(cudaStatus);

    int size = 1000;
    int bytes = size * sizeof(float);

    float *d_a, *d_b, *d_c_1, *d_c_2;
    cudaStatus = cudaMallocHost((void **)&d_a, bytes);
    CUDA_CHECK_ERROR(cudaStatus);
    cudaStatus = cudaMallocHost((void **)&d_b, bytes);
    CUDA_CHECK_ERROR(cudaStatus);
    cudaStatus = cudaMallocHost((void **)&d_c_1, bytes);
    CUDA_CHECK_ERROR(cudaStatus);
    cudaStatus = cudaMallocHost((void **)&d_c_2, bytes);
    CUDA_CHECK_ERROR(cudaStatus);

    // 初始化数据
    for (int i = 0; i < size; i++) {
        d_a[i] = static_cast<float>(i);
        d_b[i] = static_cast<float>(i * 2);
        d_c_1[i] = 0.0f;
        d_c_2[i] = 0.0f;
    }

    // 调用元素加法函数
    ElementwiseAddAsync(d_a, d_b, d_c_1, size, stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < size; i++) {
        assert(d_a[i] + d_b[i] == d_c_1[i]);
    }

    // 调用4元素加法函数
    ElementwiseAddx4Async(d_a, d_b, d_c_2, size, stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < size; i++) {
        assert(d_a[i] + d_b[i] == d_c_2[i]);
    }

    // 释放内存
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_c_1);
    cudaFreeHost(d_c_2);

    // 销毁CUDA流
    cudaStreamDestroy(stream);

    return 0;
}