#ifndef KERNELS_CUH
#define KERNELS_CUH

// C = A * B  (A: MxK, B: KxN, C: MxN), row-major
void launch_matmul(const float* d_A, const float* d_B, float* d_C,
                    int M, int K, int N);

// C = relu(A * B + bias)   bias broadcast over rows, shape [N]
// fused: matmul + bias-add + relu in a single kernel
void launch_matmul_bias_relu(const float* d_A, const float* d_B,
                              const float* d_bias, float* d_C,
                              int M, int K, int N);

// elementwise add: C = A + B
void launch_add(const float* d_A, const float* d_B, float* d_C, int n);

// elementwise relu: B = relu(A)
void launch_relu(const float* d_A, float* d_B, int n);

#endif // KERNELS_CUH