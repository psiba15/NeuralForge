#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "../cuda/kernels.cuh"
#include "../cuda/cuda_utils.h"
#include "../cuda/memory_pool.h"

void cpu_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

void run_benchmark(int M, int K, int N, const char* label) {
    printf("--- %s: (%d x %d) * (%d x %d) ---\n", label, M, K, K, N);

    size_t sizeA = (size_t)M*K*sizeof(float);
    size_t sizeB = (size_t)K*N*sizeof(float);
    size_t sizeC = (size_t)M*N*sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_cpu = (float*)malloc(sizeC);
    float* h_C_gpu = (float*)malloc(sizeC);

    for (size_t i = 0; i < (size_t)M*K; ++i) h_A[i] = (float)(rand() % 200) / 100.0f - 1.0f;
    for (size_t i = 0; i < (size_t)K*N; ++i) h_B[i] = (float)(rand() % 200) / 100.0f - 1.0f;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    GPUMemoryPool& pool = GPUMemoryPool::instance();
    float* d_A = (float*)pool.allocate(sizeA);
    float* d_B = (float*)pool.allocate(sizeB);
    float* d_C = (float*)pool.allocate(sizeC);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    launch_matmul(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    pool.deallocate(d_A);
    pool.deallocate(d_B);
    pool.deallocate(d_C);

    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)M*N; ++i) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("CPU time: %.3f ms\n", cpu_ms);
    printf("GPU time: %.3f ms (incl. memory transfer)\n", gpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);
    printf("Max diff (correctness check): %.6f\n\n", max_diff);

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

int main() {
    printf("=== Step 6: CPU vs GPU Matmul Benchmark (MNIST-scale) ===\n\n");
    srand(42);

    run_benchmark(128, 784, 64,  "Layer 1 (input -> hidden), batch 128");
    run_benchmark(128, 64,  10,  "Layer 2 (hidden -> output), batch 128");
    run_benchmark(512, 784, 128, "Larger batch, wider hidden");

    printf("=== Benchmark complete ===\n");
    return 0;
}