#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_util.h"
#include "gpu_buffer.h"
#include "dynamic_array.h"

#include <vector>

namespace hvvr {

template <typename T>
DynamicArray<T> makeDynamicArray(const GPUBuffer<T>& src) {
    DynamicArray<T> rval(src.size());
    cutilSafeCall(cudaMemcpy(rval.data(), src, src.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return rval;
}

template <typename T>
GPUBuffer<T> makeGPUBuffer(const DynamicArray<T>& src) {
    return GPUBuffer<T>(src.size(), src.data());
}

template <typename T>
GPUBuffer<T> makeGPUBuffer(const std::vector<T>& src) {
    return GPUBuffer<T>(src.size(), src.data());
}

template <typename T>
inline void safeCudaFreeHost(T*& ptrHndl) {
    if (ptrHndl) {
        cutilSafeCall(cudaFreeHost(ptrHndl));
        ptrHndl = nullptr;
    }
}

inline void safeCudaEventDestroy(cudaEvent_t& e) {
    if (e) {
        cutilSafeCall(cudaEventDestroy(e));
        e = nullptr;
    }
}

inline void safeCudaStreamDestroy(cudaStream_t& stream) {
    if (stream) {
        cutilSafeCall(cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

} // namespace hvvr
