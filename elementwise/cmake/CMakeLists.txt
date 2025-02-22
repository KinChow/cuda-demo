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