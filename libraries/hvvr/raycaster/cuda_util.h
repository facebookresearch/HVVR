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
        fprintf(stderr, "error %d: CudaSafeCall() failed at %s:%d with %s\n", error, file, line,
                cudaGetErrorString(error));
#if defined(_WIN32)
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


// Based on https://stackoverflow.com/questions/52286202/dynamic-dispatch-to-template-function-c
// Use to generate all template function permutations and dispatch properly at runtime for a prefix of template booleans
// Makes calling cuda kernels with many permutations concise.
// Example:
// Change
// if (b0) {
// 	if (b1) {
// 		if (b2) {
// 			myFunc<true, true, true, otherArgs>(args);
// 		}
// 		else {
// 			myFunc<true, true, false, otherArgs>(args);
// 		}
// 	} else {
// 		if (b2) {
// 			myFunc<true, false, true, otherArgs>(args);
// 		}
// 		else {
// 			myFunc<true, false, false, otherArgs>(args);
// 		}
// 	}
// } else {
// 	if (b1) {
// 		if (b2) {
// 			myFunc<false, true, true, otherArgs>(args);
// 		}
// 		else {
// 			myFunc<false, true, false, otherArgs>(args);
// 		}
// 	} else {
// 		if (b2) {
// 			myFunc<false, false, true, otherArgs>(args);
// 		}
// 		else {
// 			myFunc<false, false, false, otherArgs>(args);
// 		}
// 	}
// }
// into:
// std::array<bool, 3> bargs = { { b0, b1, b2 } };
// dispatch_bools<3>{}(bargs, [&](auto...Bargs) {
//     myFunc<decltype(Bargs)::value..., otherArgs>(args);
// });
//
// You may want to #pragma warning( disable : 4100) around the call, since there will be unrefenced Bargs in the call
// chain
template <bool b>
using kbool = std::integral_constant<bool, b>;

#pragma warning(push)
#pragma warning(disable : 4100)
template <std::size_t max>
struct dispatch_bools {
    template <std::size_t N, class F, class... Bools>
    void operator()(std::array<bool, N> const& input, F&& continuation, Bools...) {
        if (input[max - 1])
            dispatch_bools<max - 1>{}(input, continuation, kbool<true>{}, Bools{}...);
        else
            dispatch_bools<max - 1>{}(input, continuation, kbool<false>{}, Bools{}...);
    }
};
template <>
struct dispatch_bools<0> {
    template <std::size_t N, class F, class... Bools>
    void operator()(std::array<bool, N> const& input, F&& continuation, Bools...) {
        continuation(Bools{}...);
    }
};
#pragma warning(pop)
