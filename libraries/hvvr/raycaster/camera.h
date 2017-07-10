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
#include "graphics_types.h"
#include "foveated.h"
#include "sample_hierarchy.h"
#include "samples.h"

#include <string>


namespace hvvr {

// preprocessed samples, ready for rendering
struct SampleData {
    SampleHierarchy samples;
    uint32_t splitColorSamples = 1;
    uint32_t sampleCount;

    DynamicArray<int32_t> imageLocationToSampleIndex;
    // Flat array of sample positions (in vector2 format) without fancy swizzling for CPU vectorization
    DynamicArray<float> blockedSamplePositions;
    DynamicArray<Sample::Extents> blockedSampleExtents;

    FloatRect sampleBounds = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    uint32_t validSampleCount = 0;
    ThinLens lens = {0.0f, 5.0f};

    SampleData(){};
    SampleData(const Sample* rawSamples,
               uint32_t rawSampleCount,
               uint32_t splitColorSamples,
               const matrix3x3& sampleToCamera,
               ThinLens lens,
               uint32_t rtWidth,
               uint32_t rtHeight);
};

// TODO(anankervis): merge with GPU version of this class
class Camera {
    friend class Raycaster;
    // TODO(anankervis): remove
    friend void polarSpaceFoveatedSetup(Raycaster* raycaster);
public:
    Camera(const FloatRect& viewport, float apertureRadius);
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

    void setSampleData(const SampleData& sampleData);
    const SampleData& getSampleData() const;

    matrix3x3 getSampleToCamera() const;
    // Beware - this isn't actually suitable for taking a 2D sample coordinate + Z and converting to world space.
    // Samples can be in any arbitrary space, packing, or function we choose. What's important is that when we
    // unpack them, they turn into camera-relative 3D rays (origin offset + direction). From there, we can convert
    // into world space using cameraToWorld.
    matrix4x4 getSampleToWorld() const;
    matrix4x4 getWorldToSample() const;
    void setCameraToWorld(const transform& cameraToWorld);
    matrix4x4 getCameraToWorld() const;
    const vector3& getTranslation() const;
    vector3 getForward() const;

protected:
    // TODO(anankervis): clean up direct access of protected members by Raycaster

    matrix4x4 _previousWorldToEyeSpaceMatrix = matrix4x4::identity();
    matrix3x3 _previousEyeSpaceToPreviousSampleSpaceMatrix = matrix3x3::identity();

    // Incremeted on every render
    uint32_t _frameCount = 0;
    // Random jitters in [0,1]^2
    std::vector<vector2> _frameJitters;
    bool _newHardwareTarget = false;
    FloatRect _viewport;
    ThinLens _lens = {0.0f, 1.0f};
    bool _enabled = true;
    ImageResourceDescriptor _renderTarget;
    RaycasterOutputMode _outputMode = RaycasterOutputMode::COLOR_RGBA8;
    FoveatedSampleData _foveatedSampleData;
    // Only for polar foveated sampling
    std::vector<vector2ui> _polarRemapToPixel;

    DynamicArray<RayPacketFrustum3D> _blockFrustaTransformed;
    DynamicArray<RayPacketFrustum3D> _tileFrustaTransformed;

    transform _cameraToWorld = transform::identity();

    SampleData _sampleData;

    vector3 _eyeDir;
};

} // namespace hvvr
