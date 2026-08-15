#include <cstdio>
#include <cstdlib>
#include <cmath>
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

bool allclose(const float* a, const float* b, int n, float tol = 1e-2f) {
    for (int i = 0; i < n; ++i)
        if (fabs(a[i] - b[i]) > tol) {
            printf("Mismatch at %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    return true;
}

int main() {
    printf("=== Step 4 Tests ===\n\n");

    int M = 64, K = 32, N = 48;
    size_t sizeA = M*K*sizeof(float);
    size_t sizeB = K*N*sizeof(float);
    size_t sizeC = M*N*sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);
    float* h_bias = (float*)malloc(N*sizeof(float));

    for (int i = 0; i < M*K; ++i) h_A[i] = (float)(rand() % 100) / 50.0f - 1.0f;
    for (int i = 0; i < K*N; ++i) h_B[i] = (float)(rand() % 100) / 50.0f - 1.0f;
    for (int i = 0; i < N; ++i) h_bias[i] = (float)(rand() % 100) / 100.0f;

    // ---- Test 1: matmul correctness ----
    GPUMemoryPool& pool = GPUMemoryPool::instance();
    float* d_A = (float*)pool.allocate(sizeA);
    float* d_B = (float*)pool.allocate(sizeB);
    float* d_C = (float*)pool.allocate(sizeC);

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    launch_matmul(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    cpu_matmul(h_A, h_B, h_C_ref, M, K, N);

    bool test1 = allclose(h_C, h_C_ref, M*N);
    printf("[Test 1] Matmul kernel correctness: %s\n", test1 ? "PASS" : "FAIL");

    // ---- Test 2: fused matmul+bias+relu correctness ----
    float* d_bias = (float*)pool.allocate(N*sizeof(float));
    float* d_C2 = (float*)pool.allocate(sizeC);
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, N*sizeof(float), cudaMemcpyHostToDevice));

    launch_matmul_bias_relu(d_A, d_B, d_bias, d_C2, M, K, N);

    float* h_C2 = (float*)malloc(sizeC);
    CUDA_CHECK(cudaMemcpy(h_C2, d_C2, sizeC, cudaMemcpyDeviceToHost));

    float* h_C2_ref = (float*)malloc(sizeC);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float v = h_C_ref[i*N+j] + h_bias[j];
            h_C2_ref[i*N+j] = v > 0.0f ? v : 0.0f;
        }

    bool test2 = allclose(h_C2, h_C2_ref, M*N);
    printf("[Test 2] Fused matmul+bias+relu correctness: %s\n", test2 ? "PASS" : "FAIL");

    // ---- Test 3: memory pool reuse ----
    pool.deallocate(d_C2);
    void* reused = pool.allocate(sizeC);
    bool test3 = (reused == d_C2);
    printf("[Test 3] Memory pool block reuse: %s\n", test3 ? "PASS" : "FAIL");
    pool.deallocate(reused);

    // ---- Test 4: elementwise add + relu ----
    int n = 1000;
    float *h_x = (float*)malloc(n*sizeof(float));
    float *h_y = (float*)malloc(n*sizeof(float));
    float *h_out = (float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; ++i) { h_x[i] = i - 500; h_y[i] = 1.0f; }

    float* d_x = (float*)pool.allocate(n*sizeof(float));
    float* d_y = (float*)pool.allocate(n*sizeof(float));
    float* d_out = (float*)pool.allocate(n*sizeof(float));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, n*sizeof(float), cudaMemcpyHostToDevice));

    launch_add(d_x, d_y, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost));

    bool test4 = true;
    for (int i = 0; i < n; ++i)
        if (fabs(h_out[i] - (h_x[i] + h_y[i])) > 1e-4f) { test4 = false; break; }
    printf("[Test 4] Elementwise add kernel: %s\n", test4 ? "PASS" : "FAIL");

    launch_relu(d_x, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost));
    bool test5 = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_x[i] > 0 ? h_x[i] : 0.0f;
        if (fabs(h_out[i] - expected) > 1e-4f) { test5 = false; break; }
    }
    printf("[Test 5] Elementwise relu kernel: %s\n", test5 ? "PASS" : "FAIL");

    printf("\nActive blocks in pool: %zu\n", pool.active_block_count());
    printf("Total GPU memory allocated: %.2f KB\n", pool.total_allocated_bytes() / 1024.0f);

    pool.release_all();

    free(h_A); free(h_B); free(h_C); free(h_C_ref); free(h_bias);
    free(h_C2); free(h_C2_ref); free(h_x); free(h_y); free(h_out);

    bool all_pass = test1 && test2 && test3 && test4 && test5;
    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}