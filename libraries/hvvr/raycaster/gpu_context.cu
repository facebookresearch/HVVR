/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "cuda_raycaster.h"
#include "gpu_context.h"
#include "memory_helpers.h"

namespace hvvr {

// TODO(anankervis): remove
GPUContext* gGPUContext = nullptr;

void GPUContext::getCudaGraphicsResources(std::vector<cudaGraphicsResource_t>& resources) {
    for (const auto& c : cameras) {
        if (c.resultsResource) {
            resources.push_back(c.resultsResource);
        }
    }
}

void GPUContext::maybeMapResources() {
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

void GPUContext::unmapResources() {
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
    unmapResources();
    for (auto& c : cameras) {
        if (c.resultsResource) {
            cutilSafeCall(cudaGraphicsUnregisterResource(c.resultsResource));
        }
        c.resultImage.reset();
        c.sampleResults = GPUBuffer<uint32_t>();

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
