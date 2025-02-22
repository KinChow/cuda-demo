/*
 * @Author: Zhou Zijian 
 * @Date: 2025-02-23 00:42:33 
 * @Last Modified by:   Zhou Zijian 
 * @Last Modified time: 2025-02-23 00:42:33 
 */

#include <iostream>
#include <elementwise.h>
#include <torch/types.h>
#include <torch/extension.h>

void torch_ElementwiseAdd(torch::Tensor a, torch::Tensor b, torch::Tensor c)
{
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");

    ElementwiseAdd(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), a.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ElementwiseAdd", &torch_ElementwiseAdd, "Elementwise Add (CUDA)"); }