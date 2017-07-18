/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gpu_context.h"
#include "memory_helpers.h"

#include <cuda_profiler_api.h>


namespace hvvr {

bool GPUContext::cudaInit() {
    int deviceCount = 0;
    cutilSafeCall(cudaGetDeviceCount(&deviceCount));

    int device = 0;
#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    cudaDeviceProp deviceProps = {};

    // if we're on Windows, search for a non-TCC device
    for (int n = 0; n < deviceCount; n++) {
        cudaGetDeviceProperties(&deviceProps, n);
        if (deviceProps.tccDriver == 0) {
            device = n;
            break;
        }
    }
#endif
    cutilSafeCall(cudaSetDevice(device));

    uint32_t deviceFlags = 0;
    deviceFlags |= cudaDeviceMapHost;
    if (cudaSuccess != cudaSetDeviceFlags(deviceFlags)) {
        assert(false);
        return false;
    }

    return true;
}

void GPUContext::cudaCleanup() {
    cutilSafeCall(cudaProfilerStop()); // Flush profiling data for nvprof
}

GPUContext::GPUContext() : graphicsResourcesMapped(false) {
}

GPUContext::~GPUContext() {
    cleanup();
}

void GPUContext::getCudaGraphicsResources(std::vector<cudaGraphicsResource_t>& resources) {
    for (const auto& c : cameras) {
        if (c.resultsResource) {
            resources.push_back(c.resultsResource);
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

        for (auto& c : cameras) {
            // Assumes if the result image is a linear vector, that we are directly writing into the result resource
            if (c.resultImage.height() <= 1) {
                if (c.resultsResource) {
                    c.resultImage.updateFromLinearGraphicsResource(c.resultsResource, c.d_sampleRemap.size(),
                                                                   outputModeToPixelFormat(c.outputMode));
                }
            }
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
        if (c.resultsResource) {
            cutilSafeCall(cudaGraphicsUnregisterResource(c.resultsResource));
        }
        c.resultImage.reset();
        c.d_sampleResults = GPUBuffer<uint32_t>();

        safeCudaEventDestroy(c.transferTileToCPUEvent);
        safeCudaStreamDestroy(c.stream);
        safeCudaFreeHost(c.tileFrustaPinned);
        safeCudaFreeHost(c.cullBlockFrustaPinned);
        safeCudaFreeHost(c.foveatedWorldSpaceTileFrustaPinned);
        safeCudaFreeHost(c.foveatedWorldSpaceBlockFrustaPinned);
    }
    cameras.clear();
}

GPUCamera& GPUContext::getCreateCamera(const Camera* cameraPtr, bool& created) {
    created = false;
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (cameras[i].cameraPtr == cameraPtr) {
            return cameras[i];
        }
    }

    GPUCamera camera(cameraPtr);
    cameras.emplace_back(camera);

    created = true;
    return cameras[cameras.size() - 1];
}

} // namespace hvvr
