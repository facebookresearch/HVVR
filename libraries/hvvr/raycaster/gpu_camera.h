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
#include "foveated.h"
#include "gpu_buffer.h"
#include "gpu_image.h"
#include "gpu_samples.h"
#include "raycaster_spec.h"
#include "samples.h"
#include "texture_internal.h"
#include "vector_math.h"
#include <stdint.h>

namespace hvvr {

class Camera;
class GPUContext;
class GPUSceneState;
struct CameraBeams;
struct EccentricityToTexCoordMapping;


struct RaycasterGBufferSubsample {
    uint32_t triIndex;
    uint32_t sampleMask;

    template <uint32_t AARate>
    CUDA_HOST_DEVICE static constexpr uint32_t getSampleMaskAll() {
        return (AARate < 32) ? ((uint32_t(1) << AARate) - 1) : ~uint32_t(0);
    }
};

// Keep in sync with blockcull.h/RayPacketFrustum2D
struct GPURayPacketFrustum {
    float xMin;
    float xNegMax;
    float yMin;
    float yNegMax;
};

struct TemporalFilterSettings {
    // 1 entirely use current sample
    float alpha = .05f;
    float stddevMultiplier = 1.0f;
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
    bool enable;
    float f_e;
};

inline PixelFormat outputModeToPixelFormat(RaycasterOutputFormat mode) {
    (void)mode;
    return PixelFormat::RGBA8_SRGB;
}

#pragma warning(push)
#pragma warning(disable : 4324) // structure was padded due to alignment specifier

// TODO(anankervis): merge with Camera class
class GPUCamera {
public:
    // Single point-of-origin rays defined in "batch-space" which will usually be camera-space
    // In a standard foveated setup, these would be in eye-space
    GPUBuffer<DirectionalBeam> d_batchSpaceBeams;
    ThinLens lens;
    // how far to jitter the samples this frame. length() < 1
    vector2 frameJitter = {0.0f, 0.0f};

    // Intermediate output, between intersection and resolve
    GPUBuffer<RaycasterGBufferSubsample> d_gBuffer;

    GPUBuffer<int32_t> d_sampleRemap;
    uint32_t splitColorSamples = 1;

    // Final result after mapping sample results to screen space
    GPUImage resultImage;

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

    GPUBuffer<uint32_t> d_sampleResults;
    GPUBuffer<uint32_t> d_sampleResultsRemapped;

    uint32_t sampleCount;
    GPUBuffer<float> d_tMaxBuffer;

    //// Foveation-specific data
    GPUBuffer<vector2ui> d_polarRemapToPixel;
    struct PolarTextures {
        Texture2D raw;
        Texture2D depth;

        Texture2D moment1;
        Texture2D moment2;
    } polarTextures;

    // Terminology from http://cwyman.org/papers/siga16_gazeTrackedFoveatedRendering.pdf
    TemporalFilterSettings temporalFilterSettings;
    Texture2D previousResultTexture;
    Texture2D resultTexture;

    ContrastEnhancementBuffers contrastEnhancementBuffers;
    ContrastEnhancementSettings contrastEnhancementSettings;

    EccentricityMap eccentricityMap;
    float maxEccentricityRadians;
    ////

    cudaGraphicsResource_t resultsResource = NULL;
    uint32_t validSampleCount = 0;
    cudaStream_t stream = 0;

    RaycasterOutputFormat outputMode = RaycasterOutputFormat::COLOR_RGBA8;

    // Used as a unique ID in the GPUContext
    const Camera* cameraPtr;

    GPUCamera(const Camera* cameraPtr);

    void initLookupTables(int MSAARate);

    // per-frame updates
    void setCameraJitter(vector2 jitter);

    // sample config updates
    void updateConfig(RaycasterOutputFormat _outputMode,
                      int32_t* sampleRemap,
                      DirectionalBeam* directionalBeams,
                      ThinLens _lens,
                      uint32_t _sampleCount,
                      uint32_t imageWidth,
                      uint32_t imageHeight,
                      uint32_t imageStride,
                      uint32_t _splitColorSamples);

    // attach a texture from a 3D API
    bool bindTexture(GPUContext& gpuContext, ImageResourceDescriptor texture);
    // GPUContext::graphicsResourcesMapped must be true before calling this function
    void copyImageToBoundTexture();

    void copyImageToCPU(ImageResourceDescriptor cpuTarget);

    void intersectShadeResolve(GPUSceneState& sceneState, const matrix4x4& cameraToWorld);

    // convert from linear post-resolve results buffer to results image
    void remap();

    // Foveation-specific
    void registerPolarFoveatedSamples(const std::vector<vector2ui>& polarRemapToPixel,
                                      float _maxEccentricityRadians,
                                      const EccentricityMap& eccentricityMap,
                                      uint32_t samplesPerRing,
                                      uint32_t paddedSampleCount);
    void updateEyeSpaceFoveatedSamples(BeamBatch& beamHierarchy);
    void remapPolarFoveated();
    void foveatedPolarToScreenSpace(const matrix4x4& eyeToEyePrevious,
                                    const matrix3x3& eyePreviousToSamplePrevious,
                                    const matrix3x3& sampleToEye);

    // dump the current set of rays (including subsamples) to a CPU buffer
    void dumpRays(std::vector<SimpleRay>& rays, bool outputScanlineOrder, const matrix4x4& cameraToWorld);

protected:
    // intersect triangles
    void intersect(GPUSceneState& sceneState, const CameraBeams& cameraBeams, const matrix4x4& cameraToWorld);

    // fill empty tiles with default clear value
    void clearEmpty();
    // shade occupied tiles and resolve AA subsamples
    void shadeAndResolve(GPUSceneState& sceneState, const CameraBeams& cameraBeams, const matrix4x4& cameraToWorld);

    void getEccentricityMap(EccentricityToTexCoordMapping& map) const;
};
#pragma warning(pop)

} // namespace hvvr
