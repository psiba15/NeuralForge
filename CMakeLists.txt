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
