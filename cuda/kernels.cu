#include "kernels.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

// ---------- Tiled matmul kernel ----------
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------- Fused matmul + bias + relu kernel ----------
__global__ void matmul_bias_relu_kernel(const float* A, const float* B,
                                         const float* bias, float* C,
                                         int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + bias[col];
        C[row * N + col] = val > 0.0f ? val : 0.0f;
    }
}

// ---------- Elementwise kernels ----------
__global__ void add_kernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

__global__ void relu_kernel(const float* A, float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) B[idx] = A[idx] > 0.0f ? A[idx] : 0.0f;
}

// ---------- Launch wrappers ----------
void launch_matmul(const float* d_A, const float* d_B, float* d_C,
                    int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_matmul_bias_relu(const float* d_A, const float* d_B,
                              const float* d_bias, float* d_C,
                              int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_bias_relu_kernel<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, K, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_add(const float* d_A, const float* d_B, float* d_C, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_relu(const float* d_A, float* d_B, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_A, d_B, n);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}