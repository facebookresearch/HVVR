/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "constants_math.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "gpu_samples.h"
#include "kernel_constants.h"
#include "magic_constants.h"
#include "memory_helpers.h"

#include <functional>
#include <random>

#ifdef DX_SUPPORTED
#include <cuda_d3d11_interop.h>
#else
struct ID3D11Texture2D;
struct ID3D11Buffer;
#endif

#include <cuda_gl_interop.h>


namespace hvvr {

uint32_t pixelFormatSize(PixelFormat pixelFormat) {
    switch (pixelFormat) {
        case PixelFormat::RGBA8_SRGB:
            return 4;
        case PixelFormat::RGBA32F:
            return 16;
        default:
            assert(false);
            return 0;
    }
}

void Camera_StreamedData::reset(uint32_t tileCount) {
    tileCountOccupied = 0;
    tileCountEmpty = 0;

    tileIndexRemapEmpty.resizeDestructive(tileCount);
    tileIndexRemapOccupied.resizeDestructive(tileCount);
    tileTriRanges.resizeDestructive(tileCount);
    triIndices.resizeDestructive(MAX_TRI_INDICES_TO_INTERSECT);
    tileFrusta3D.resizeDestructive(tileCount);
}

GPUCamera::GPUCamera(const Camera* cameraPtr) : streamedIndexCPU(0), streamedIndexGPU(-1), cameraPtr(cameraPtr) {
    cutilSafeCall(cudaStreamCreate(&stream));

    for (int n = 0; n < frameBuffering; n++) {
        cutilSafeCall(cudaEventCreateWithFlags(&streamed[n].gpuDone, cudaEventBlockingSync | cudaEventDisableTiming));
    }
}
// TODO: there's no cleanup code for GPUCamera, yet, and it would be a big pain to clean it up to properly support
// the full set of constructors and assignments (especially move variants) given the number of members...

void GPUCamera::initLookupTables(int _MSAARate) {
    // getSubsampleUnitOffset needs a compile-time constant for MSAARate
    enum { MSAARate = COLOR_MODE_MSAA_RATE };
    assert(MSAARate == _MSAARate);

    std::uniform_real_distribution<float> uniformRandomDist(0.0f, 1.0f);
    std::mt19937 generator;
    auto r = std::bind(uniformRandomDist, std::ref(generator));

    // lookup table for random lens position
    enum { TileCount = DOF_LENS_POS_LOOKUP_TABLE_TILES };
    std::vector<vector2> tileSubsampleLensPosData(TILE_SIZE * TileCount * MSAARate);
    for (int tile = 0; tile < TileCount; tile++) {
        for (int sample = 0; sample < int(TILE_SIZE); sample++) {
            float rotation = r() * Tau;

            for (int subsample = 0; subsample < MSAARate; subsample++) {
                vector2 pos =
                    getSubsampleUnitOffset<MSAARate>(vector2(0.0f, 0.0f), (subsample * 7 + 7) % MSAARate, rotation);

                // tileSubsampleLensPosData[tile * TILE_SIZE * MSAARate + subsample * TILE_SIZE + sample].x =
                //    uint32_t(floatToHalf(pos.x)) | (uint32_t(floatToHalf(pos.y)) << 16);
                tileSubsampleLensPosData[tile * TILE_SIZE * MSAARate + subsample * TILE_SIZE + sample] = pos;
            }
        }
    }
    d_tileSubsampleLensPos.resizeDestructive(TILE_SIZE * TileCount * MSAARate);
    d_tileSubsampleLensPos.upload(tileSubsampleLensPosData.data());
}

Camera_StreamedData* GPUCamera::streamedDataLock(uint32_t tileCount) {
    Camera_StreamedData* rval = streamed + streamedIndexCPU;
    cutilSafeCall(cudaEventSynchronize(rval->gpuDone));

    streamedIndexCPU = (streamedIndexCPU + 1) % frameBuffering;

    rval->reset(tileCount);
    return rval;
}

void GPUCamera::streamedDataUnlock() {
    streamedIndexGPU = (streamedIndexGPU + 1) % frameBuffering;

    Camera_StreamedData* streamSrc = streamed + streamedIndexGPU;

    // some things don't have appropriate access patterns for reasonable PCIe streaming perf, so we copy them
    local.tileIndexRemapEmpty.resizeDestructive(streamSrc->tileIndexRemapEmpty.size());
    local.tileIndexRemapEmpty.uploadAsync(streamSrc->tileIndexRemapEmpty.data(), stream);

    local.tileIndexRemapOccupied.resizeDestructive(streamSrc->tileIndexRemapOccupied.size());
    local.tileIndexRemapOccupied.uploadAsync(streamSrc->tileIndexRemapOccupied.data(), stream);

    cutilFlush(stream);

    local.tileTriRanges.resizeDestructive(streamSrc->tileTriRanges.size());
    local.tileTriRanges.uploadAsync(streamSrc->tileTriRanges.data(), stream);

    local.tileFrusta3D.resizeDestructive(streamSrc->tileFrusta3D.size());
    local.tileFrusta3D.uploadAsync(streamSrc->tileFrusta3D.data(), stream);

    cutilFlush(stream);
}

void GPUCamera::streamedDataGpuDone() {
    cutilSafeCall(cudaEventRecord(streamed[streamedIndexGPU].gpuDone, stream));
    cutilFlush(stream);
}

void GPUCamera::setCameraJitter(vector2 jitter) {
    frameJitter.x = jitter.x;
    frameJitter.y = jitter.y;
}

void GPUCamera::updatePerFrame(vector3 cameraPos,
                               vector3 cameraLookVector,
                               const matrix3x3& _sampleToCamera,
                               const matrix4x4& _cameraToWorld) {
    position = vector3(cameraPos.x, cameraPos.y, cameraPos.z);
    lookVector = vector3(cameraLookVector.x, cameraLookVector.y, cameraLookVector.z);
    sampleToCamera = _sampleToCamera;
    cameraToWorld = _cameraToWorld;
}

static int getMSAARate(RaycasterOutputMode outputMode) {
    return (outputMode == RaycasterOutputMode::COLOR_RGBA8) ? COLOR_MODE_MSAA_RATE : 1;
}

static TextureFormat pixelFormatToTextureFormat(PixelFormat format) {
    switch (format) {
        case PixelFormat::RGBA8_SRGB:
            return TextureFormat::r8g8b8a8_unorm_srgb;
        case PixelFormat::RGBA32F:
            return TextureFormat::r32g32b32a32_float;
        default:
            assert(false);
    }
    return TextureFormat::none;
}

// TODO(anankervis): merge the different functions that duplicate camera resource creation
void GPUCamera::updateConfig(RaycasterOutputMode _outputMode,
                             int32_t* sampleRemap,
                             float* sampleLocations,
                             Sample::Extents* sampleExtents,
                             ThinLens _lens,
                             uint32_t _sampleCount,
                             uint32_t imageWidth,
                             uint32_t imageHeight,
                             uint32_t imageStride,
                             uint32_t _splitColorSamples) {
    // one sample per output pixel, one sample per pentile subpixel, or one sample per R,G,B channel
    assert(splitColorSamples == 1 || splitColorSamples == 2 || splitColorSamples == 3);
    splitColorSamples = _splitColorSamples;
    validSampleCount = imageWidth * imageHeight * splitColorSamples;
    d_sampleRemap = GPUBuffer<int32_t>(sampleRemap, sampleRemap + validSampleCount);
    sampleCount = _sampleCount;
    d_sampleLocations = GPUBuffer<vector2>((vector2*)sampleLocations, (vector2*)(sampleLocations) + sampleCount);
    d_sampleExtents =
        GPUBuffer<Sample::Extents>((Sample::Extents*)sampleExtents, (Sample::Extents*)(sampleExtents) + sampleCount);

    outputMode = _outputMode;
    int msaaRate = getMSAARate(outputMode);
    d_gBuffer = GPUBuffer<RaycasterGBufferSubsample>(sampleCount * msaaRate);

    PixelFormat outputFormat = outputModeToPixelFormat(outputMode);
    TextureFormat textureFormat = pixelFormatToTextureFormat(outputFormat);

    previousResultTexture =
        createEmptyTexture(imageWidth, imageHeight, textureFormat, cudaAddressModeClamp, cudaAddressModeClamp);
    resultTexture =
        createEmptyTexture(imageWidth, imageHeight, textureFormat, cudaAddressModeClamp, cudaAddressModeClamp);

    contrastEnhancementSettings.enable = true;
    contrastEnhancementSettings.f_e = 0.2f;
    contrastEnhancementBuffers.horizontallyFiltered =
        createEmptyTexture(imageWidth, imageHeight, textureFormat, cudaAddressModeClamp, cudaAddressModeClamp);
    contrastEnhancementBuffers.fullyFiltered =
        createEmptyTexture(imageWidth, imageHeight, textureFormat, cudaAddressModeClamp, cudaAddressModeClamp);

    auto pixelFormat = outputModeToPixelFormat(outputMode);
    d_sampleResults =
        GPUBuffer<uint32_t>((sampleCount * pixelFormatSize(pixelFormat) + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    resultImage.update(imageWidth, imageHeight, imageStride, pixelFormat);
    lens = _lens;

    initLookupTables(msaaRate);
}

void GPUCamera::registerPolarFoveatedSamples(const std::vector<vector2ui>& polarRemapToPixel,
                                             float _maxEccentricityRadians,
                                             const std::vector<float>& ringEccentricities,
                                             const std::vector<float>& eccentricityCoordinateMap,
                                             uint32_t samplesPerRing,
                                             uint32_t paddedSampleCount) {
    PixelFormat outputFormat = outputModeToPixelFormat(outputMode);
    sampleCount = paddedSampleCount;
    d_sampleResults = GPUBuffer<uint32_t>((paddedSampleCount * pixelFormatSize(outputFormat) + sizeof(uint32_t) - 1) /
                                          sizeof(uint32_t));
    d_sampleLocations = GPUBuffer<vector2>(paddedSampleCount);
    d_sampleExtents = GPUBuffer<Sample::Extents>(paddedSampleCount);
    d_sampleRemap = GPUBuffer<int32_t>(paddedSampleCount);

    // For temporal filtering
    d_tMaxBuffer = GPUBuffer<float>(paddedSampleCount);

    maxEccentricityRadians = _maxEccentricityRadians;
    d_eccentricityCoordinateMap = makeGPUBuffer(eccentricityCoordinateMap);
    d_ringEccentricities = makeGPUBuffer(ringEccentricities);

    int msaaRate = getMSAARate(outputMode);
    size_t totalSubsampleCount = paddedSampleCount * msaaRate;

    // Allow us to launch a complete tile
    d_gBuffer = GPUBuffer<RaycasterGBufferSubsample>(totalSubsampleCount);

    d_polarRemapToPixel = makeGPUBuffer(polarRemapToPixel);

    TextureFormat textureFormat = pixelFormatToTextureFormat(outputFormat);

    polarFoveatedImage = createEmptyTexture(samplesPerRing, uint32_t(polarRemapToPixel.size() / samplesPerRing),
                                            textureFormat, cudaAddressModeWrap, cudaAddressModeClamp);
    previousPolarFoveatedImage = createEmptyTexture(samplesPerRing, uint32_t(polarRemapToPixel.size() / samplesPerRing),
                                                    textureFormat, cudaAddressModeWrap, cudaAddressModeClamp);
    rawPolarFoveatedImage = createEmptyTexture(samplesPerRing, uint32_t(polarRemapToPixel.size() / samplesPerRing),
                                               textureFormat, cudaAddressModeWrap, cudaAddressModeClamp, false);
    polarFoveatedDepthImage =
        createEmptyTexture(samplesPerRing, uint32_t(polarRemapToPixel.size() / samplesPerRing),
                           TextureFormat::r32_float, cudaAddressModeWrap, cudaAddressModeClamp, false);

    initLookupTables(msaaRate);
}

bool GPUCamera::bindTexture(GPUContext& gpuContext, ImageResourceDescriptor texture) {
    if (resultsResource) {
        gpuContext.interopUnmapResources();
        cutilSafeCall(cudaGraphicsUnregisterResource(resultsResource));
        resultsResource = nullptr;
    }
#ifdef DX_SUPPORTED
    if (texture.memoryType == ImageResourceDescriptor::MemoryType::DX_TEXTURE) {
        // cudaGraphicsRegisterFlagsNone is only valid flag as of 7/22/2016
        cutilSafeCall(cudaGraphicsD3D11RegisterResource(&resultsResource, (ID3D11Texture2D*)texture.data,
                                                        cudaGraphicsRegisterFlagsNone));
    }
#endif
    if (texture.memoryType == ImageResourceDescriptor::MemoryType::OPENGL_TEXTURE) {
        cutilSafeCall(cudaGraphicsGLRegisterImage(&resultsResource, (GLuint)(uint64_t)texture.data, GL_TEXTURE_2D,
                                                  cudaGraphicsMapFlagsWriteDiscard));
    }

    return true;
}

void GPUCamera::copyImageToBoundTexture() {
    cudaArray* cuArray;
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&cuArray, resultsResource, 0, 0));
    size_t srcStride = resultImage.width() * resultImage.bytesPerPixel(); // tightly packed
    cutilSafeCall(cudaMemcpy2DToArrayAsync(cuArray, 0, 0, resultImage.data(), srcStride, srcStride,
                                           resultImage.height(), cudaMemcpyDeviceToDevice, stream));
}

void GPUCamera::copyImageToCPU(uint32_t* imageData, uint32_t imageWidth, uint32_t imageHeight, uint32_t imageStride) {
    auto pixFormat = outputModeToPixelFormat(outputMode);
    resultImage.update(imageWidth, imageHeight, imageStride, pixFormat);

    cutilSafeCall(
        cudaMemcpyAsync(imageData, resultImage.data(), resultImage.sizeInMemory(), cudaMemcpyDeviceToHost, 0));
}

void GPUCamera::acquireTileCullData(SimpleRayFrustum* tileFrusta, SimpleRayFrustum* blockFrusta) {
    cutilSafeCall(cudaEventSynchronize(transferTileToCPUEvent));

    size_t blockCount = d_cullBlockFrusta.size();
    memcpy(blockFrusta, foveatedWorldSpaceBlockFrustaPinned, sizeof(SimpleRayFrustum) * blockCount);
    memcpy(tileFrusta, foveatedWorldSpaceTileFrustaPinned, sizeof(SimpleRayFrustum) * blockCount * TILES_PER_BLOCK);
}

void GPUCamera::intersectShadeResolve(GPUSceneState& sceneState) {
    Camera_StreamedData& streamedData = streamed[streamedIndexGPU];

    // prep the scene
    sceneState.update();
    cutilSafeCall(cudaStreamWaitEvent(stream, sceneState.updateEvent, 0));

    // The intersect and resolve kernels assume every thread will map to a valid work item, with valid input and output
    // slots. Sample count should be padded to a minimum of CUDA_GROUP_SIZE. In practice, it is padded to BLOCK_SIZE.
    assert(sampleCount % CUDA_GROUP_SIZE == 0);

    if (streamedData.tileCountEmpty > 0) {
        clearEmpty();
    }

    SampleInfo sampleInfo(*this);
    if (streamedData.tileCountOccupied > 0) {
        intersect(sceneState, sampleInfo);
        shadeAndResolve(sceneState, sampleInfo);
    }

    streamedDataGpuDone();
}

} // namespace hvvr
