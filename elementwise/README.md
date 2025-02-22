# elementwise

## 算法

**elementwise算子**的功能是对输入张量（一个或多个）进行**逐元素独立运算**，即对张量中每个对应位置的元素执行相同的操作，同时支持形状兼容（如广播机制）。常见的操作包括加减乘除、激活函数（如ReLU、Sigmoid）等，其核心特点是不改变张量的维度结构（或通过广播自动扩展维度）。



**elementwise-add**是elementwise算子的一种具体实现，特指**逐元素加法**。其行为如下：

1. **输入要求**：两个张量形状相同，或可通过广播机制兼容（例如，将形状为 `(3,1)` 和 `(1,4)` 的张量扩展为 `(3,4)`）。
2. **计算规则**：对每个对应位置的元素执行加法运算，即输出张量满足 `C[i,j,...] = A[i,j,...] + B[i,j,...]`。
3. **应用场景**：常见于神经网络中的残差连接（ResNet）、特征融合、参数更新等需要张量相加的操作。



**示例**：
若输入张量 `A = [[1, 2], [3, 4]]` 和 `B = [[5, 6], [7, 8]]`，则 `elementwise-add` 的输出为 `[[6, 8], [10, 12]]`。若 `B` 是标量 `5`，广播后结果为 `[[6, 7], [8, 9]]`。



**意义**：

- **计算高效**：逐元素操作可高度并行化，适合GPU加速。
- **灵活性**：通过广播机制支持不同形状张量的运算，简化了模型设计（如偏置项的添加）。





## CUDA

### 主机代码

#### 内核执行

CUDA内核启动语法 `<<<grid_dim, block_dim, shared_mem_size, stream>>>` 用于配置内核在GPU上的执行方式，具体参数功能如下：

1. `grid_dim`
   - **功能**：定义网格（Grid）的维度，即包含多少个线程块（Block）。
   - **类型**：可以是 `int` 或 `dim3`（支持三维维度）。
   - **示例**：
     - `grid_dim = 4` 表示网格包含 4 个一维线程块。
     - `dim3 grid_dim(2, 3)` 表示网格包含 `2×3=6` 个二维线程块。
2. `block_dim`
   - **功能**：定义每个线程块（Block）的维度，即包含多少个线程（Thread）。
   - **类型**：可以是 `int` 或 `dim3`（支持三维维度）。
   - **示例**：
     - `block_dim = 256` 表示每个线程块包含 256 个一维线程。
     - `dim3 block_dim(16, 16)` 表示每个线程块包含 `16×16=256` 个二维线程。
3. `shared_mem_size`（可选）
   - **功能**：为每个线程块动态分配共享内存（Shared Memory）的大小（单位：字节）。
   - **默认值**：`0`（不分配）。
   - **示例**：
     - `shared_mem_size = 1024` 表示每个线程块分配 1024 字节的共享内存。
4. `stream`（可选）
   - **功能**：指定内核在哪个 CUDA 流（Stream）中异步执行。
   - **默认值**：`0`（默认流，同步执行）。





### 内核代码

```
// 元素加法的CUDA核函数
__global__ void ElementwiseAddKernel(float *a, float *b, float *c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) {
        return;
    }
    c[idx] = a[idx] + b[idx];
}

// 定义FLOAT4宏，将float指针转换为float4指针
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// 4元素加法的CUDA核函数
__global__ void ElementwiseAddx4Kernel(float *a, float *b, float *c, int size) {
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
```

在 `ElementwiseAddx4Kernel` 中，通过 **向量化内存访问（float4）** 和 **减少线程数量** 实现了对 `ElementwiseAddKernel` 的优化，具体优化点和效果如下：

* **向量化内存访问（float4）**

  - **合并内存事务**：
    使用 `float4`（4个连续的 `float`）替代单个 `float` 的读写，将 **4次独立内存访问合并为1次**。CUDA 全局内存的访存模式以 32 字节/128 字节为基本单位，合并访问可减少内存事务数量，提升带宽利用率。

  - **对齐访问**：
    `float4` 要求内存地址 16 字节对齐（每个 `float` 4 字节，4×4=16），对齐访问可避免非对齐内存操作的开销。

* **减少线程数量**

  - **更高的计算/内存比**：
    每个线程处理 **4个元素**，线程总数减少到原来的 1/4，降低了线程调度开销（如线程块调度、寄存器分配）。

  - **隐藏内存延迟**：
    单线程处理更多数据时，可通过指令级并行性（ILP）掩盖内存访问延迟。

* **减少全局索引计算**
  - 索引计算 `idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4` 代替逐个计算，减少了全局地址计算的次数。