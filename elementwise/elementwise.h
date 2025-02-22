/*
 * @Author: Zhou Zijian 
 * @Date: 2025-02-23 00:42:49 
 * @Last Modified by:   Zhou Zijian 
 * @Last Modified time: 2025-02-23 00:42:49 
 */

#ifndef ELEMENTWISE_H
#define ELEMENTWISE_H

#include <cuda_runtime.h>

void ElementwiseAdd(float *a, float *b, float *c, int size);

void ElementwiseAddx4(float *a, float *b, float *c, int size);

void ElementwiseAddAsync(float *a, float *b, float *c, int size, cudaStream_t stream);

void ElementwiseAddx4Async(float *a, float *b, float *c, int size, cudaStream_t stream);

#endif  // ELEMENTWISE_H