#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "cuda_decl.h"
#include "gbuffer.h"
#include "gpu_buffer.h"
#include "gpu_image.h"
#include "raycaster_spec.h"
#include "samples.h"
#include "texture_internal.h"
#include "vector_math.h"
#include "foveated.h"

namespace hvvr {

class Camera;
class GPUContext;
class GPUSceneState;
struct SampleInfo;
struct EccentricityToTexCoordMapping;

// Keep in sync with blockcull.h/RayPacketFrustum
struct GPURayPacketFrustum {
    float xMin;
    float xNegMax;
    float yMin;
    float yNegMax;
};

struct TemporalFilterSettings {
    // 1 entirely use current sample
    float alpha = 0.1f;
    float stddevMultiplier = 4.0f;
    bool inPolarSpace = false;
};

struct TileTriRange {
    uint32_t start;
    uint32_t end;
};

// data which is streamed from the CPU to GPU (CPU writes once, GPU consumes once)
struct Camera_StreamedData {
    cudaEvent_t gpuDone; // is the GPU done consuming the data?

    uint32_t tileCountOccupied;
    uint32_t tileCountEmpty;

    // maps compacted tile index to original tile index
    GPUBufferHostWC<uint32_t> tileIndexRemapEmpty;
    GPUBufferHostWC<uint32_t> tileIndexRemapOccupied;

    GPUBufferHostWC<TileTriRange> tileTriRanges;
    GPUBufferHostWC<SimpleRayFrustum> tileFrusta3D;

    GPUBufferHostWC<uint32_t> triIndices;

    void reset(uint32_t tileCount);
};

// data in local memory on the GPU
struct Camera_LocalData {
    GPUBuffer<uint32_t> tileIndexRemapEmpty;
    GPUBuffer<uint32_t> tileIndexRemapOccupied;

    GPUBuffer<TileTriRange> tileTriRanges;
    GPUBuffer<SimpleRayFrustum> tileFrusta3D;
};

struct ContrastEnhancementBuffers {
    // For NVIDIA-style contrast enhancement
    Texture2D horizontallyFiltered;
    Texture2D fullyFiltered;
};

struct ContrastEnhancementSettings {
    bool enable = false;
    float f_e;
};

inline PixelFormat outputModeToPixelFormat(RaycasterOutputMode mode) {
    (void)mode;
    return PixelFormat::RGBA8_SRGB;
}

#pragma warning(push)
#pragma warning(disable : 4324) // structure was padded due to alignment specifier

// TODO(anankervis): merge with Camera class
class GPUCamera {
public:
    // Terminology from http://cwyman.org/papers/siga16_gazeTrackedFoveatedRendering.pdf
    TemporalFilterSettings temporalFilterSettings;

    GPUBuffer<RaycasterGBufferSubsample> d_gBuffer;
    GPUBuffer<vector2> d_sampleLocations;
    GPUBuffer<Sample::Extents> d_sampleExtents;
    GPUBuffer<int32_t> d_sampleRemap;
    uint32_t splitColorSamples = 1;

    Camera_LocalData local;

    // how many copies of streamed data to prevent stalls?
    // 2 = double buffered
    // 3 = triple buffered
    // beware latency at 3+
    enum { frameBuffering = 2 };
    Camera_StreamedData streamed[frameBuffering];
    int streamedIndexCPU; // the next Camera_StreamedData available for filling by the CPU
    int streamedIndexGPU; // the next Camera_StreamedData to be consumed by the GPU

    Camera_StreamedData* streamedDataLock(uint32_t tileCount);
    void streamedDataUnlock();
    // the GPU is done consuming streamed data for this frame
    void streamedDataGpuDone();

    GPUBuffer<vector2> d_tileSubsampleLensPos;

    GPUBuffer<vector2ui> d_polarRemapToPixel;

    GPUBuffer<PrecomputedDirectionSample> d_foveatedDirectionalSamples;
    GPUBuffer<SimpleRayFrustum> d_foveatedEyeSpaceTileFrusta;
    GPUBuffer<SimpleRayFrustum> d_foveatedEyeSpaceBlockFrusta;
    GPUBuffer<SimpleRayFrustum> d_foveatedWorldSpaceTileFrusta;
    GPUBuffer<SimpleRayFrustum> d_foveatedWorldSpaceBlockFrusta;
    SimpleRayFrustum* foveatedWorldSpaceTileFrustaPinned = nullptr;
    SimpleRayFrustum* foveatedWorldSpaceBlockFrustaPinned = nullptr;

    GPUBuffer<GPURayPacketFrustum> d_tileFrusta;
    GPUBuffer<GPURayPacketFrustum> d_cullBlockFrusta;
    GPURayPacketFrustum* tileFrustaPinned = nullptr;
    GPURayPacketFrustum* cullBlockFrustaPinned = nullptr;

    GPUBuffer<float> d_ringEccentricities;

    cudaEvent_t transferTileToCPUEvent = nullptr;

    GPUBuffer<uint32_t> d_sampleResults;
    uint32_t sampleCount;
    GPUBuffer<float> d_tMaxBuffer;
    GPUImage resultImage;
    // For polarFoveatedReconstruction
    Texture2D rawPolarFoveatedImage;
    Texture2D previousPolarFoveatedImage;
    Texture2D polarFoveatedImage;
    Texture2D polarFoveatedDepthImage;

    // For temporal filtering in polarFoveatedReconstruction
    Texture2D previousResultTexture;
    Texture2D resultTexture;

    ContrastEnhancementBuffers contrastEnhancementBuffers;
    ContrastEnhancementSettings contrastEnhancementSettings;

    float maxEccentricityRadians;
    GPUBuffer<float> d_eccentricityCoordinateMap;

    cudaGraphicsResource_t resultsResource = NULL;
    uint32_t validSampleCount = 0;

    vector3 position;
    vector3 lookVector;
    matrix3x3 sampleToCamera;
    matrix4x4 cameraToWorld;

    ThinLens lens;

    cudaStream_t stream = 0;

    RaycasterOutputMode outputMode = RaycasterOutputMode::COLOR_RGBA8;

    // how far to jitter the samples this frame. length() < 1
    vector2 frameJitter = {0.0f, 0.0f};

    const Camera* cameraPtr;

    GPUCamera(const Camera* cameraPtr);

    void initLookupTables(int MSAARate);

    // per-frame updates
    void setCameraJitter(vector2 jitter);
    void updatePerFrame(vector3 cameraPos,
                        vector3 cameraLookVector,
                        const matrix3x3& _sampleToCamera,
                        const matrix4x4& _cameraToWorld);
    void updatePerFrameFoveatedData(const FloatRect& sampleBounds,
                                    const matrix3x3& cameraToSample,
                                    const matrix3x3& eyeToCamera,
                                    const matrix4x4& eyeToWorld);

    // sample config updates
    void updateConfig(RaycasterOutputMode _outputMode,
                      int32_t* sampleRemap,
                      float* sampleLocations,
                      Sample::Extents* sampleExtents,
                      ThinLens _lens,
                      uint32_t _sampleCount,
                      uint32_t imageWidth,
                      uint32_t imageHeight,
                      uint32_t imageStride,
                      uint32_t _splitColorSamples);
    void registerPolarFoveatedSamples(const std::vector<vector2ui>& polarRemapToPixel,
                                      float _maxEccentricityRadians,
                                      const std::vector<float>& ringEccentricities,
                                      const std::vector<float>& eccentricityCoordinateMap,
                                      uint32_t samplesPerRing,
                                      uint32_t paddedSampleCount);
    void updateEyeSpaceFoveatedSamples(const ArrayView<PrecomputedDirectionSample> precomputedDirectionalSamples);

    // attach a texture from a 3D API, used when OUTPUT_MODE = OUTPUT_MODE_3D_API
    bool bindTexture(GPUContext& gpuContext, ImageResourceDescriptor texture);
    // GPUContext::graphicsResourcesMapped must be true before calling this function
    void copyImageToBoundTexture();

    void copyImageToCPU(uint32_t* imageData, uint32_t imageWidth, uint32_t imageHeight, uint32_t imageStride);

    // copy GPU-generated frusta data back to CPU for traversal (foveated render path)
    // Assumes CalculatePerFrameFoveatedData has previously been called
    void acquireTileCullData(SimpleRayFrustum* tileFrusta, SimpleRayFrustum* blockFrusta);

    void intersectShadeResolve(GPUSceneState& sceneState);

    // convert from linear post-resolve results buffer to results image
    void remap();
    void remapPolarFoveated();

    void foveatedPolarToScreenSpace(const matrix4x4& eyeToEyePrevious,
                                    const matrix3x3& eyePreviousToSamplePrevious,
                                    const matrix3x3& sampleToEye);

    // dump the current set of rays (including subsamples) to a CPU buffer
    void dumpRays(std::vector<SimpleRay>& rays, bool outputScanlineOrder);

protected:

    // intersect triangles
    void intersect(GPUSceneState& sceneState, const SampleInfo& sampleInfo);

    // fill empty tiles with default clear value
    void clearEmpty();
    // shade occupied tiles and resolve AA subsamples
    void shadeAndResolve(GPUSceneState& sceneState, const SampleInfo& sampleInfo);

    void getEccentricityMap(EccentricityToTexCoordMapping& map) const;
    void foveatedPolarTemporalFilter(const matrix4x4& eyeToEyePrevious);
};
#pragma warning(pop)

} // namespace hvvr
