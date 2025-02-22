# elementwise-cmake

## CUDA

### 主机代码

#### 异步执行流（`cudaStream_t`）

##### 功能

**异步执行流**主要功能是管理 GPU 操作的并行性和顺序性。通过多个流（stream），可以实现以下功能：

1. **并行执行**：在支持硬件多队列的 GPU 上，不同流中的核函数（kernel）、内存拷贝（H2D/D2H）等操作可以并发执行。
2. **任务隔离**：将独立的任务分配到不同流中，避免默认流（NULL stream）的顺序执行限制。
3. **流水线优化**：通过重叠计算和数据传输（如计算与内存拷贝同时进行），提高 GPU 利用率。



##### 调用方法

```c++
// 创建流
cudaStream_t stream;
cudaStreamCreate(&stream);  // 创建流

// 在流中执行操作
kernel<<<grid, block, 0, stream>>>(args...); // 启动核函数并指定流
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream); // 异步内存拷贝

// 同步流
cudaStreamSynchronize(stream);  // 阻塞主机，直到流中所有操作完成

// 销毁流
cudaStreamDestroy(stream);  // 释放流资源
```



##### 优缺点

* 优点

  * 性能提升：
    * 计算与传输重叠：通过异步操作，可以在执行核函数的同时进行数据拷贝（需使用固定内存）。
    * 多核函数并行：在支持多流的 GPU 上，不同流中的核函数可以并发执行。

  * 灵活性：适用于流水线设计（如深度学习推理中的多批次处理）。

* 缺点

  * 复杂性增加：
    * 需要手动管理流之间的依赖关系，避免数据竞争（如不同流操作同一内存区域）。
    * 错误使用可能导致死锁或结果错误。
  * 资源限制：
    * GPU 的硬件多队列数量有限（如 NVIDIA Ampere 架构支持 128 个流），超出后无法真正并发。
    * 每个流需要额外的内存和计算资源管理。
  * 调试困难：异步操作使得调试和性能分析更加复杂。





#### 内存管理（`cudaMallocHost`和`cudaFreeHost`）

##### 功能

1. **`cudaMallocHost`**

   - **功能**：在主机（CPU）内存中分配 **页锁定内存（Pinned Memory）**。

   - **特点**：

     - 内存页被锁定在物理内存中，不可被操作系统交换到磁盘（不可分页）。
     - 支持 GPU 直接通过 DMA（直接内存访问）与主机内存高效交互。

   - **语法**：

     ```c
     cudaError_t cudaMallocHost(void **ptr, size_t size);
     ```

2. **`cudaFreeHost`**

   - **功能**：释放由 `cudaMallocHost` 分配的页锁定内存。

   - **语法**：

     ```c
     cudaError_t cudaFreeHost(void *ptr);
     ```



##### 使用场景

1. **高速主机-设备数据传输**
   - 当需要频繁在主机和设备之间拷贝数据时（如深度学习训练中的批量数据传输），使用页锁定内存可减少传输延迟。
2. **异步内存操作**
   - 与 `cudaMemcpyAsync` 结合使用，实现数据传输与计算的并行（需配合 CUDA Stream）。
3. **零拷贝内存（Zero-Copy）**
   - 允许 GPU 直接访问主机内存，避免显式拷贝（适用于设备内存不足或数据复用场景）。
4. **流式处理**
   - 实时数据处理（如视频流、传感器数据流）中，页锁定内存支持高效流水线操作。



##### 优缺点

* 优点
  * 传输速度快
    * DMA 直接访问页锁定内存，避免了普通可分页内存的额外拷贝开销（省去“锁页+拷贝”步骤）。
  * 支持异步操作
    * 与 `cudaMemcpyAsync` 和 CUDA Stream 结合，实现计算与传输的重叠。
  * 设备直接访问
    * 通过零拷贝技术（`cudaHostAlloc` + `cudaHostGetDevicePointer`），GPU 可直接读写主机内存。
* 缺点
  * 内存资源占用高
    * 页锁定内存占用物理内存空间，过量分配可能导致系统内存不足（影响其他程序运行）。
  * 分配/释放开销大
    * 分配页锁定内存的时间比普通 `malloc` 更长，频繁操作可能影响性能。
  * 系统稳定性风险
    * 过量锁定内存可能触发操作系统限制（如 Linux 的 `ulimit`），导致程序崩溃。
  * 平台兼容性
    * 某些嵌入式系统或旧操作系统可能不支持页锁定内存。





## CMake

```
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
set(PROJECT_NAME elementwise)

# 设置CUDA架构（在project命令之前）
# 获取计算能力 nvidia-smi --query-gpu=compute_cap --format=csv
set(CMAKE_CUDA_ARCHITECTURES "86")

# 定义项目
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# 查找CUDA包
find_package(CUDA REQUIRED)

# 指定头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

# 指定CUDA源文件
set(CUDA_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../elementwise.cu)

# 指定C++源文件
set(CPP_SOURCE main.cpp)

# 添加可执行文件
add_executable(${PROJECT_NAME} ${CUDA_SOURCE} ${CPP_SOURCE})

# 设置目标属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 包含CUDA头文件
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

# 链接CUDA库
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
```



### 1. 设置 GPU 计算架构（Compute Capability）

```cmake
set(CMAKE_CUDA_ARCHITECTURES "xx")
```

- **功能**：指定目标 GPU 的计算能力（Compute Capability），例如 `80` 对应 Ampere 架构（如 A100），`75` 对应 Turing 架构（如 RTX 2080）。
- **用户需替换**：将 `"xx"` 替换为实际值（通过 `nvidia-smi --query-gpu=compute_cap --format=csv` 查询实际 GPU 的计算能力）。
- **意义**：确保生成的 CUDA 代码能适配目标 GPU 的硬件特性，避免兼容性问题。



### 2. 定义项目并启用语言支持

```cmake
project(${PROJECT_NAME} LANGUAGES CXX CUDA)
```

- **功能**：声明项目名称及支持的语言（C++ 和 CUDA）。
- **参数解析**：
  - `${PROJECT_NAME}`：项目名称。
  - `LANGUAGES CXX CUDA`：启用 C++ 和 CUDA 编译支持，允许混合使用 `.cpp` 和 `.cu` 文件。



### 3. 查找 CUDA 工具包

```cmake
find_package(CUDA REQUIRED)
```

- **功能**：在系统中搜索 CUDA 开发环境（包括编译器 `nvcc`、库文件、头文件等）。
- **关键参数**：`REQUIRED` 表示如果找不到 CUDA，则终止构建过程并报错。
- **输出变量**：此命令会定义 `CUDA_INCLUDE_DIRS`（CUDA 头文件路径）和 `CUDA_LIBRARIES`（CUDA 库文件路径）等变量供后续使用。



### 4. 生成可执行文件

```cmake
add_executable(${PROJECT_NAME} ${CUDA_SOURCE} ${CPP_SOURCE})
```

- **功能**：将 CUDA 源文件编译为可执行文件。
- **隐含行为**：CMake 会自动调用 `nvcc` 编译器处理 `.cu` 文件。



### 5. 启用 CUDA 可分离编译

```cmake
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
```

- **功能**：启用 CUDA 可分离编译模式，允许将设备代码（Device Code）分多个步骤编译，最终链接为完整二进制。
- **意义**：
  - 减少重复编译时间（适用于多设备代码文件）。
  - 解决复杂模板或大规模设备代码的编译问题。



### 6. 包含 CUDA 头文件目录

```cmake
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
```

- **功能**：添加 CUDA 头文件（如 `cuda_runtime.h`）的搜索路径到编译器的头文件包含路径中。
- **作用域**：`PRIVATE` 表示仅对目标生效，不影响其他目标。
- **变量解析**：`${CUDA_INCLUDE_DIRS}` 由 `find_package(CUDA)` 自动填充，通常指向 `/usr/local/cuda/include`。



### 7. 链接 CUDA 库

```cmake
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
```

- **功能**：将可执行文件与 CUDA 运行时库（如 `cudart`）链接。
- **作用域**：`PRIVATE` 表示仅当前目标依赖这些库。
- **变量解析**：`${CUDA_LIBRARIES}` 由 `find_package(CUDA)` 自动填充，包含必要的 CUDA 库路径。