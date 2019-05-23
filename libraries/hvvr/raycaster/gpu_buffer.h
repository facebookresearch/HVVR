#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_util.h"


// Wrapper for a GPU buffer of POD types
template <typename T, bool PinnedHostMemory = false, bool WriteCombined = false>
class GPUBuffer {
public:
    ~GPUBuffer() {
        free();
    }

    GPUBuffer() : _data(nullptr), _dataHost(nullptr), _size(0), _capacity(0) {}
    GPUBuffer(size_t newSize) : GPUBuffer() {
        resizeDestructive(newSize);
    }
    GPUBuffer(size_t newSize, uint8_t fillPattern) : GPUBuffer() {
        resizeDestructive(newSize);
        memset(fillPattern);
    }
    GPUBuffer(size_t newSize, const T* cpuData) : GPUBuffer() {
        resizeDestructive(newSize);
        cutilSafeCall(cudaMemcpy(_data, cpuData, sizeof(T) * _size, cudaMemcpyDefault));
    }
    GPUBuffer(const T* cpuDataStart, const T* cpuDataEnd) : GPUBuffer() {
        size_t newSize = cpuDataEnd - cpuDataStart;
        *this = GPUBuffer<T>(newSize, cpuDataStart);
    }

    GPUBuffer<T>& operator=(const GPUBuffer<T>& src) {
        resizeDestructive(src._size);
        cutilSafeCall(cudaMemcpy(_data, src._data, sizeof(T) * _size, cudaMemcpyDefault));
        return *this;
    }
    GPUBuffer(const GPUBuffer<T>& src) : GPUBuffer() {
        *this = src;
    }

    GPUBuffer<T>& operator=(GPUBuffer<T>&& src) {
        if (this != &src) {
            this->~GPUBuffer();

            _data = std::move(src._data);
            _dataHost = std::move(src._dataHost);
            _size = std::move(src._size);
            _capacity = std::move(src._capacity);

            src._data = nullptr;
            src._dataHost = nullptr;
            src._size = 0;
            src._capacity = 0;
        }
        return *this;
    }
    GPUBuffer(GPUBuffer<T>&& src) : GPUBuffer() {
        *this = std::move(src);
    }

    // Will never shrink the capacity
    // Existing contents are not preserved
    void resizeDestructive(size_t newSize) {
        if (newSize > _capacity) {
            alloc(newSize);
        }
        _size = newSize;
    }

    void memset(uint8_t fillPattern = 0) {
        cutilSafeCall(cudaMemset(_data, fillPattern, _size * sizeof(T)));
    }
    void memsetAsync(uint8_t fillPattern, cudaStream_t stream) {
        cutilSafeCall(cudaMemsetAsync(_data, fillPattern, _size * sizeof(T), stream));
    }

    void upload(const T* cpuData) {
        cutilSafeCall(cudaMemcpy(_data, cpuData, sizeof(T) * _size, cudaMemcpyDefault));
    }
    void uploadAsync(const T* cpuData, cudaStream_t stream) {
        cutilSafeCall(cudaMemcpyAsync(_data, cpuData, sizeof(T) * _size, cudaMemcpyDefault, stream));
    }
    void uploadAsync(const T* cpuData, size_t dstOffset, size_t count, cudaStream_t stream) {
        cutilSafeCall(cudaMemcpyAsync(_data + dstOffset, cpuData, sizeof(T) * count, cudaMemcpyDefault, stream));
    }

    void readback(T* cpuDst) {
        cutilSafeCall(cudaMemcpy(cpuDst, _data, sizeof(T) * _size, cudaMemcpyDefault));
    }
    void readbackAsync(T* cpuDst, cudaStream_t stream) {
        cutilSafeCall(cudaMemcpyAsync(cpuDst, _data, sizeof(T) * _size, cudaMemcpyDefault, stream));
    }

    T* data() const {
        return _data;
    }
    operator T*() const {
        return data();
    }

    // CPU-visible pointer for pinned memory
    T* dataHost() const {
        return _dataHost;
    }

    size_t size() const {
        return _size;
    }
    size_t capacity() const {
        return _capacity;
    }

protected:
    T* _data;
    T* _dataHost;
    size_t _size;
    size_t _capacity;

    void alloc(size_t count) {
        free();

        if (PinnedHostMemory) {
            uint32_t flags = cudaHostAllocMapped;
            if (WriteCombined) {
                flags |= cudaHostAllocWriteCombined;
            }
            cutilSafeCall(cudaHostAlloc(&_dataHost, count * sizeof(T), flags));
            cutilSafeCall(cudaHostGetDevicePointer(&_data, _dataHost, 0));
        } else {
            cutilSafeCall(cudaMalloc(&_data, count * sizeof(T)));
            _dataHost = nullptr;
        }
        _capacity = count;
    }

    void free() {
        if (_data) {
            if (PinnedHostMemory) {
                cutilSafeCall(cudaFreeHost(_dataHost));
            } else {
                cutilSafeCall(cudaFree(_data));
            }
            _data = nullptr;
            _dataHost = nullptr;
        }
        _size = 0;
        _capacity = 0;
    }
};

// data lives on the host and is cached
template <typename T>
using GPUBufferHost = GPUBuffer<T, true, false>;

// data lives on the host and is write-combined (don't attempt to read it from the CPU)
template <typename T>
using GPUBufferHostWC = GPUBuffer<T, true, true>;
