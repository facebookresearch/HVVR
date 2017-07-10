/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "cuda_raycaster.h"
#include "cuda_util.h"
#include "gpu_buffer.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "gpu_image.h"
#include "intersect.h"
#include "kernel_constants.h"
#include "memory_helpers.h"
#include "remap.h"
#include "resolve.h"
#include "shading_helpers.h"
#include "texture.h"

#include <cuda_profiler_api.h>


namespace hvvr {

void AcquireTileCullInformation(Camera* cameraPtr,
                                SimpleRayFrustum* tileFrusta,
                                SimpleRayFrustum* blockFrusta) {
    bool created;
    auto& camera = gGPUContext->getCreateCamera(cameraPtr, created);
    assert(!created);
    size_t blockCount = camera.d_cullBlockFrusta.size();

    cutilSafeCall(cudaEventSynchronize(camera.transferTileToCPUEvent));
    // memcpy(blockFrusta, (void*)camera.cullBlockFrustaPinned, sizeof(GPURayPacketFrustum) * blockCount);
    // memcpy(tileFrusta, (void*)camera.tileFrustaPinned, sizeof(GPURayPacketFrustum) * blockCount * TILES_PER_BLOCK);

    memcpy(blockFrusta, (void*)camera.foveatedWorldSpaceBlockFrustaPinned, sizeof(SimpleRayFrustum) * blockCount);
    memcpy(tileFrusta, (void*)camera.foveatedWorldSpaceTileFrustaPinned,
           sizeof(SimpleRayFrustum) * blockCount * TILES_PER_BLOCK);
}

struct tonemap_functor {
    tonemap_functor() {}
    CUDA_DEVICE uint32_t operator()(const vector4& c) const {
        return ToColor4Unorm8SRgb(ACESFilm(c));
    }
};

int CopyImageToBoundTexture(Camera* cameraPtr) {
    assert(gGPUContext->graphicsResourcesMapped);
    bool created;
    auto& camera = gGPUContext->getCreateCamera(cameraPtr, created);

    auto d_data = camera.resultImage.data();
    cudaArray* cuArray;
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&cuArray, camera.resultsResource, 0, 0));
    size_t srcStride = camera.resultImage.width() * camera.resultImage.bytesPerPixel(); // tightly packed
    cutilSafeCall(cudaMemcpy2DToArrayAsync(cuArray, 0, 0, d_data, srcStride, srcStride, camera.resultImage.height(),
                                           cudaMemcpyDeviceToDevice, camera.stream));
    /*
    // For debugging vector4 outputs onto a 32-bit buffer
    size_t pixCount = camera.resultImage.stride() * camera.resultImage.height();
    static GPUBuffer<uint32_t> tonemappedResult(pixCount);
    thrust::transform(thrust::device, (vector4*)d_data, ((vector4*)d_data) + pixCount, tonemappedResult.begin(),
                  tonemap_functor());
    cudaArray* cuArray;
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&cuArray, camera.resultsResource, 0, 0));
    cutilSafeCall(cudaMemcpy2DToArrayAsync(cuArray, 0, 0, tonemappedResult, // d_data,
                                       camera.resultImage.stride() * 4, // camera.resultImage.bytesPerPixel(),
                                       camera.resultImage.width() * 4,  // camera.resultImage.bytesPerPixel(),
                                       camera.resultImage.height(), cudaMemcpyDeviceToDevice, camera.stream));
    */
    return 0;
}

void DeferredMSAAIntersect(Camera* cameraPtr) {
#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    assert(gGPUContext->graphicsResourcesMapped);
#endif

    GPUSceneState& sceneState = gGPUContext->sceneState;

    bool created;
    auto& camera = gGPUContext->getCreateCamera(cameraPtr, created);
    assert(!created);

    Camera_StreamedData& streamed = camera.streamed[camera.streamedIndexGPU];

    // prep the scene
    sceneState.update();
    cutilSafeCall(cudaStreamWaitEvent(camera.stream, sceneState.updateEvent, 0));

    // The intersect and resolve kernels assume every thread will map to a valid work item, with valid input and output
    // slots. Sample count should be padded to a minimum of CUDA_BLOCK_SIZE. In practice, it is padded to BLOCK_SIZE.
    assert(camera.sampleCount % CUDA_BLOCK_SIZE == 0);

    if (streamed.tileCountEmpty > 0) {
        ClearEmptyTiles(camera, camera.sampleResults, camera.stream);
    }

    SampleInfo sampleInfo(camera);
    if (streamed.tileCountOccupied > 0) {
        DeferredIntersectTiles(camera, sampleInfo, camera.sampleToCamera, camera.cameraToWorld);
        DeferredMSAAResolve(camera, camera.sampleResults, sampleInfo, camera.sampleToCamera, camera.cameraToWorld);
    }
}

int DeferredMSAAIntersectAndRemap(Camera* cameraPtr) {
#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    assert(gGPUContext->graphicsResourcesMapped);
#endif

    bool created;
    auto& camera = gGPUContext->getCreateCamera(cameraPtr, created);
    assert(!created);
    DeferredMSAAIntersect(cameraPtr);
    RemapSampleResultsToImage(camera);

    cutilSafeCall(cudaEventRecord(camera.streamed[camera.streamedIndexGPU].gpuDone, camera.stream));

    cutilFlush(camera.stream);

    return 0;
}

int UnmapGraphicsResources() {
    gGPUContext->unmapResources();
    return 0;
}

int MapGraphicsResources() {
    gGPUContext->maybeMapResources();
    return 0;
}

void CreateExplicitRayBuffer(Camera* cameraPtr, std::vector<SimpleRay>& rays) {
    bool created;
    auto& camera = gGPUContext->getCreateCamera(cameraPtr, created);
    assert(!created);
    uint32_t rayCount = COLOR_MODE_MSAA_RATE * camera.sampleCount;
    rays.resize(rayCount);
    GPUBuffer<SimpleRay> d_rays;
    d_rays.resizeDestructive(rayCount);
    GenerateExplicitRayBuffer(camera, d_rays.data());
    d_rays.readback(rays.data());
}

bool Init() {
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

int Cleanup() {
    gGPUContext->cleanup();
    delete gGPUContext;
    gGPUContext = nullptr;

    cutilSafeCall(cudaProfilerStop()); // Flush profiling data for nvprof

    return 0;
}

} // namespace hvvr
