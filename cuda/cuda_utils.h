#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d\n", __FILE__, __LINE__);   \
            fprintf(stderr, "  Code: %d, Reason: %s\n",                     \
                    err, cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_LAST()                                                   \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Kernel Launch Error at %s:%d\n",          \
                    __FILE__, __LINE__);                                    \
            fprintf(stderr, "  Code: %d, Reason: %s\n",                     \
                    err, cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#endif // CUDA_UTILS_H