/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gpu_context.h"
#include "memory_helpers.h"

#include <cuda_profiler_api.h>


namespace hvvr {

bool GPUContext::cudaInit(bool forceNonTCC) {
    int deviceCount = 0;
    cutilSafeCall(cudaGetDeviceCount(&deviceCount));

    int device = 0;

    if (forceNonTCC) {
        cudaDeviceProp deviceProps = {};

        // if we're on Windows, search for a non-TCC device
        for (int n = 0; n < deviceCount; n++) {
            cudaGetDeviceProperties(&deviceProps, n);
            if (deviceProps.tccDriver == 0) {
                device = n;
                break;
            }
        }
    }
    cutilSafeCall(cudaSetDevice(device));

    uint32_t deviceFlags = 0;
    deviceFlags |= cudaDeviceMapHost;
    auto error = cudaSetDeviceFlags(deviceFlags);
    if (cudaSuccess != error) {
        fprintf(stderr, "error %d: cuda call failed with %s\n", error, cudaGetErrorString(error));
        assert(false);
        return false;
    }

    return true;
}

void GPUContext::cudaCleanup() {
    cutilSafeCall(cudaProfilerStop()); // Flush profiling data for nvprof
}

GPUContext::GPUContext() : graphicsResourcesMapped(false) {}

GPUContext::~GPUContext() {
    cleanup();
}

void GPUContext::getCudaGraphicsResources(std::vector<cudaGraphicsResource_t>& resources) {
    for (const auto& c : cameras) {
        if (c->resultsResource) {
            resources.push_back(c->resultsResource);
        }
    }
}

void GPUContext::interopMapResources() {
    if (!graphicsResourcesMapped) {
        std::vector<cudaGraphicsResource_t> resources;
        getCudaGraphicsResources(resources);
        cudaStream_t stream = 0;
        if (resources.size() > 0) {
            cutilSafeCall(cudaGraphicsMapResources((int)resources.size(), resources.data(), stream));
        }
        graphicsResourcesMapped = true;
    }
}

void GPUContext::interopUnmapResources() {
    if (graphicsResourcesMapped) {
        std::vector<cudaGraphicsResource_t> resources;
        getCudaGraphicsResources(resources);
        cudaStream_t stream = 0;
        if (resources.size() > 0) {
            cutilSafeCall(cudaGraphicsUnmapResources((int)resources.size(), resources.data(), stream));
        }
        graphicsResourcesMapped = false;
    }
}

void GPUContext::cleanup() {
    interopUnmapResources();
    for (auto& c : cameras) {
        if (c->resultsResource) {
            cutilSafeCall(cudaGraphicsUnregisterResource(c->resultsResource));
        }
        c->resultImage.reset();
        c->d_sampleResults = GPUBuffer<uint32_t>();

        safeCudaStreamDestroy(c->stream);
    }
    cameras.clear();
}

GPUCamera* GPUContext::getCreateCamera(const Camera* cameraPtr, bool& created) {
    created = false;
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (cameras[i]->cameraPtr == cameraPtr) {
            return cameras[i].get();
        }
    }

    cameras.emplace_back(std::make_unique<GPUCamera>(cameraPtr));

    created = true;
    return (cameras.end() - 1)->get();
}

} // namespace hvvr
