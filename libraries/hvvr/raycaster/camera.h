#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dynamic_array.h"
#include "foveated.h"
#include "gpu_samples.h"
#include "graphics_types.h"
#include "sample_hierarchy.h"
#include "samples.h"

#include <string>


namespace hvvr {

class GPUCamera;
class GPUContext;


// preprocessed samples, ready for rendering
struct SampleData {
    BeamBatch2D samples2D;
    Sample2Dto3DMappingSettings settings2DTo3D;
    BeamBatch samples;
    uint32_t splitColorSamples = 1;

    DynamicArray<int32_t> imageLocationToSampleIndex;

    FloatRect sampleBounds = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    uint32_t validSampleCount = 0;

    SampleData(){};
    SampleData(const Sample* rawSamples,
               uint32_t rawSampleCount,
               uint32_t splitColorSamples,
               Sample2Dto3DMappingSettings settings2DTo3D,
               uint32_t rtWidth,
               uint32_t rtHeight);
    void generate3Dfrom2D(Sample2Dto3DMappingSettings settings);
};


// TODO(anankervis): merge with GPU version of this class
class Camera {
    friend class Raycaster;
    // TODO(anankervis): remove
    friend void polarSpaceFoveatedSetup(Raycaster* raycaster);

public:
    struct CPUHierarchy {
        DynamicArray<Frustum> _blockFrusta;
        DynamicArray<Frustum> _tileFrusta;
    };

    Camera(const FloatRect& viewport, float apertureRadius, GPUContext& gpuContext);
    ~Camera();

    Camera(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera& operator=(Camera&&) = delete;

    void setEnabled(bool enabled);
    bool getEnabled() const;

    // for foveated
    void setEyeDir(const vector3& eyeDir);
    const vector3& getEyeDir() const;

    void setViewport(const FloatRect& viewport);
    const FloatRect& getViewport() const;

    // For DoF sampling
    void setLens(const ThinLens& lens);
    const ThinLens& getLens() const {
        return _lens;
    }
    void setAperture(float radius);
    float getAperture() const;
    void setFocalDepth(float depth);

    void setRenderTarget(const ImageResourceDescriptor& newRenderTarget);
    void setSamples(const Sample* rawSamples, uint32_t rawSampleCount, uint32_t splitColorSamples);

    // If called with nonzero values, this camera uses a spherical section for ray generation
    // (instead of the standard perspective transform).
    void setSphericalWarpSettings(float fovXDegrees, float fovYDegrees);

    void setSampleData(const SampleData& sampleData);
    const SampleData& getSampleData() const;
    const uint32_t getSampleCount() const;

    matrix3x3 getSampleToCamera() const;

    void setCameraToWorld(const transform& cameraToWorld);
    matrix4x4 getCameraToWorld() const;
    const vector3& getTranslation() const;

    void setupRenderTarget(GPUContext& context);
    void extractImage();

    void advanceJitter();

protected:
    Sample2Dto3DMappingSettings get2DSampleMappingSettings() const;

    float _fovXDegrees = 0.0f;
    float _fovYDegrees = 0.0f;

    // TODO(anankervis): clean up direct access of protected members by Raycaster
    GPUCamera* _gpuCamera;

    // Initialize to an invalid transform since there is no previous frame on the initial frame
    matrix4x4 _worldToEyePrevious = matrix4x4::zero();
    matrix3x3 _eyePreviousToSamplePrevious = matrix3x3::identity();

    // Incremeted on every render
    uint32_t _frameCount = 0;
    // Random jitters in [0,1]^2
    std::vector<vector2> _frameJitters;
    bool _newHardwareTarget = false;
    FloatRect _viewport;
    ThinLens _lens = {0.0f, 1.0f};
    bool _enabled = true;
    ImageResourceDescriptor _renderTarget;
    RaycasterOutputFormat _outputFormat = RaycasterOutputFormat::COLOR_RGBA8;
    FoveatedSampleData _foveatedSampleData;

    // Only for polar foveated sampling
    std::vector<vector2ui> _polarRemapToPixel;

    CPUHierarchy _cpuHierarchy;

    transform _cameraToWorld = transform::identity();

    SampleData _sampleData;

    vector3 _eyeDir;
};

} // namespace hvvr
