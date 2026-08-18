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

# step 5 ke liye test executable
add_executable(test_step5
    tests/test_step5.cpp
    src/value.cpp
    src/layers.cpp
)
target_include_directories(test_step5 PRIVATE include)

# step 6 CPU part ke liye test executable
add_executable(train_mnist_cpu
    tests/train_mnist_cpu.cpp
    src/value.cpp
    src/layers.cpp
    src/mnist_loader.cpp
)
target_include_directories(train_mnist_cpu PRIVATE include)

# ══════════════════════════════════════════════════════════
# CUDA — optional. Agar CUDA compiler mile (Kaggle GPU) tabhi
# ye targets banenge. Local Windows (no GPU) pe automatically
# skip ho jayenge, koi error nahi aayega.
# ══════════════════════════════════════════════════════════
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)   # T4 GPU on Kaggle = compute capability 7.5

    message(STATUS "CUDA compiler found — building GPU targets too.")

    add_executable(test_step4
        tests/test_step4.cu
        cuda/kernels.cu
    )
    target_include_directories(test_step4 PRIVATE cuda include)
    set_target_properties(test_step4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(benchmark_mnist_matmul
        tests/benchmark_mnist_matmul.cu
        cuda/kernels.cu
    )
    target_include_directories(benchmark_mnist_matmul PRIVATE cuda include)
    set_target_properties(benchmark_mnist_matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "No CUDA compiler found — skipping GPU targets (test_step4, benchmark_mnist_matmul).")
endif()