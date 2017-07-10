#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// Very basic error handling
#define cutilSafeCall(error) __cudaSafeCall(error, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t error, const char* file, const int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "error: CudaSafeCall() failed at %s:%d with %s\n", file, line, cudaGetErrorString(error));
#ifdef _WIN32
        __debugbreak();
#else
        exit(error);
#endif
    }
}

// trigger a flush of buffered commands to kick off the GPU (mitigates latency under the WDDM driver)
inline void cutilFlush(cudaStream_t stream) {
    cudaError_t queryResult = cudaStreamQuery(stream);
    if (queryResult != cudaSuccess && queryResult != cudaErrorNotReady) {
        cutilSafeCall(queryResult); // forward the async error
    }
}

// Helper struct for generating 1D or 2D CUDA grid dimensions from block sizes and thread counts
// Could likely slightly improve performance by transforming kernels to use grid-stride loops instead
struct KernelDim {
    dim3 grid;
    dim3 block;

    KernelDim() : grid(dim3(1, 1, 1)), block(dim3(1, 1, 1)) {}

    KernelDim(size_t width, size_t height, uint32_t blockWidth, uint32_t blockHeight) {
        grid = dim3(((unsigned int)((width + blockWidth - 1) / blockWidth)),
                    (unsigned int)((height + blockHeight - 1) / blockHeight), 1);
        block = dim3(blockWidth, blockHeight, 1);
    }

    KernelDim(size_t size, uint32_t blockSize) {
        grid = dim3(((unsigned int)((size + blockSize - 1) / blockSize)), 1, 1);
        block = dim3(blockSize, 1, 1);
    }
};

#define CUDA_INF __int_as_float(0x7f800000)