#ifndef CUDA_RUNTIME_ERROR_CHECK_H
#define CUDA_RUNTIME_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
  {                                                                       \
    const cudaError_t error = call;                                       \
    if (error != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);         \
      fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
    }                                                                     \
  }

#endif // CUDA_RUNTIME_ERROR_CHECK_H
