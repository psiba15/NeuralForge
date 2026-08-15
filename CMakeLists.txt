cmake_minimum_required(VERSION 3.18)
project(NeuralForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include folder taaki #include "tensor.h" kaam kare
include_directories(include)

# executable banao
add_executable(neuralforge
    main.cpp
    src/tensor.cpp
    src/value.cpp
    src/neural.cpp
)

# CUDA Step 4 mein enable karenge — Kaggle pe
# project(NeuralForge LANGUAGES CXX CUDA)
# set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)   # T4 GPU on Kaggle = compute capability 7.5

add_executable(test_step4
    tests/test_step4.cu
    cuda/kernels.cu
)
target_include_directories(test_step4 PRIVATE cuda include)
set_target_properties(test_step4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)