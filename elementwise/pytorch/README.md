# elementwise-pytorch

使用PyTorch运行自定义CUDA算子（以elementwise-add为例）





## 环境依赖

### 安装环境

```
# 安装 Python 开发头文件
sudo apt-get install python3-dev

# 安装编译工具链
sudo apt-get install build-essential ninja-build

# 安装PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装 CUDA 工具包（可选）
sudo apt-get install nvidia-cuda-toolkit
```



### 验证环境

```
# 检查 Python.h 是否存在
find /usr/include -name "Python.h" 2>/dev/null

# 检查 ninja
ninja --version # ninja版本

# 检查 CUDA 是否可用
nvcc --version  # CUDA 版本

# 检查 PyTorch 安装
import torch
print(torch.__version__)          # PyTorch 版本
print(torch.cuda.is_available())  # CUDA 是否可用
```





## 自定义PyTorch算子

### 注册

#### 步骤

1. 编写C++/CUDA代码

   - 实现CUDA核函数和主机函数。

   - 编写PyTorch包装函数，处理张量检查和数据转换。

2. 使用Pybind11绑定

   - 通过 `PYBIND11_MODULE` 宏将C++函数暴露给Python。

   - 模块名 `TORCH_EXTENSION_NAME` 由PyTorch编译系统自动定义。



#### 说明

##### PyTorch包装函数 `torch_ElementwiseAdd`

```cpp
void torch_ElementwiseAdd(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    // 检查输入张量是否在CUDA设备上
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be a CUDA tensor");
    
    // 检查张量内存是否连续
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    
    // 调用CUDA主机函数
    ElementwiseAdd(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        a.numel()
    );
}
```

- **功能**：将PyTorch张量转换为CUDA指针并调用主机函数。
- **关键操作**：
  - **设备检查**：确保输入张量位于GPU。
  - **连续性检查**：保证张量内存布局连续（避免跨步访问）。
  - **数据指针获取**：通过 `data_ptr<T>()` 获取显存指针。
  - **元素数量获取**：`a.numel()` 获取总元素数。



##### 注册自定义算子 `PYBIND11_MODULE`

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ElementwiseAdd", &torch_ElementwiseAdd, "Elementwise Add (CUDA)");
}
```

- **功能**：将C++函数 `torch_ElementwiseAdd` 注册为Python可调用的PyTorch算子。
- **参数**：
  - `"ElementwiseAdd"`：Python中调用的函数名。
  - `&torch_ElementwiseAdd`：C++函数指针。
  - `"Elementwise Add (CUDA)"`：函数描述（可选）。





### 使用

#### 步骤

1. 编译扩展
   * 使用 `torch.utils.cpp_extension.load` 编译代码
2. 在Python中调用



#### 说明

##### 在 Python 中动态编译并加载扩展

使用 `torch.utils.cpp_extension.load` 直接编译和加载 CUDA 代码：

```python
from torch.utils.cpp_extension import load

# 动态加载 CUDA 扩展
lib = load(
    name='elementwise_lib',  # 扩展名称（任意）
    sources=['../elementwise.cu', "lib.cpp"],  # CUDA 源文件路径
    extra_include_paths=['../'],  # 头文件路径
    extra_cflags=['-std=c++17'],  # 编译选项（例如启用 C++17）
    verbose=True  # 显示编译日志（可选）
)
```



`torch.utils.cpp_extension.load` 是 PyTorch 提供的工具函数，用于 **动态编译并加载 C++/CUDA 扩展**。它简化了自定义算子的开发流程，无需手动编写 `setup.py` 或 CMake 配置，特别适合快速开发和调试。

|        参数名         |     类型      |                             作用                             |                          示例                           |
| :-------------------: | :-----------: | :----------------------------------------------------------: | :-----------------------------------------------------: |
|        `name`         |     `str`     |         **扩展模块的名称**（内部标识符，不影响调用）         |                  `name='my_cuda_lib'`                   |
|       `sources`       |  `List[str]`  |          **源文件列表**（支持 `.cpp`、`.cu` 文件）           |           `sources=['kernel.cpp', 'ops.cu']`            |
|    `extra_cflags`     |  `List[str]`  |           **C++ 编译选项**（传递给 `g++`/`clang`）           |      `extra_cflags=['-O3', '-I/path/to/include']`       |
|  `extra_cuda_cflags`  |  `List[str]`  |              **CUDA 编译选项**（传递给 `nvcc`）              |           `extra_cuda_cflags=['-arch=sm_80']`           |
|    `extra_ldflags`    |  `List[str]`  |       **链接器选项**（如动态库路径 `-L` 和库名 `-l`）        | `extra_ldflags=['-L/usr/local/cuda/lib64', '-lcudart']` |
| `extra_include_paths` |  `List[str]`  |         **头文件搜索路径**（等效于 `-I`，但更简洁）          |    `extra_include_paths=['/usr/local/cuda/include']`    |
|   `build_directory`   |     `str`     |          **编译临时文件存放目录**（默认在临时目录）          |               `build_directory='./build'`               |
|       `verbose`       |    `bool`     |            **是否显示编译日志**（调试时建议启用）            |                     `verbose=True`                      |
|      `with_cuda`      |    `bool`     | **是否启用 CUDA 支持**（若 `sources` 含 `.cu` 文件，默认自动启用） |          `with_cuda=True`（通常无需显式指定）           |
|  `is_python_module`   |    `bool`     |    **是否生成 Python 模块**（默认 `True`，保持默认即可）     |                 `is_python_module=True`                 |
|      `use_ninja`      |    `bool`     | **是否使用 Ninja 加速编译**（默认 `True`，需已安装 `ninja`） |          `use_ninja=False`（强制使用 `make`）           |
|    `undef_macros`     |  `List[str]`  |               **取消定义的宏**（等效于 `-U`）                |                `undef_macros=['NDEBUG']`                |
|    `define_macros`    | `List[Tuple]` |                  **定义宏**（等效于 `-D`）                   |  `define_macros=[('DEBUG', '1'), ('USE_FP16', None)]`   |



##### 调用自定义算子

```python
lib.ElementwiseAdd(a, b, c)  # 直接调用注册的函数名
```

- 函数名 `ElementwiseAdd` 需与 `PYBIND11_MODULE` 注册的名称一致。