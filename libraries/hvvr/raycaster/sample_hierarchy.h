#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gpu_samples.h"
#include "graphics_types.h"
#include "samples.h"
#include "traversal.h"

namespace hvvr {

struct SampleHierarchy2D {
    DynamicArray<RayPacketFrustum2D> tileFrusta;
    DynamicArray<RayPacketFrustum2D> blockFrusta;
    DynamicArray<UnpackedSample> samples;
    SampleHierarchy2D() {}
    SampleHierarchy2D(ArrayView<SortedSample> sortedSamples,
                      uint32_t blockCount,
                      uint32_t validSampleCount,
                      const FloatRect& cullRect,
                      ThinLens thinLens,
                      const matrix3x3& sampleToCamera);
};

struct Sample2Dto3DMappingSettings {
    matrix3x3 sampleToCamera;
    ThinLens thinLens;
    enum class MappingType { Perspective, SphericalSection };
    MappingType type = MappingType::Perspective;
    // Only used for SphericalSection mapping
    float fovXDegrees;
    float fovYDegrees;
    Sample2Dto3DMappingSettings() {}
    Sample2Dto3DMappingSettings(matrix3x3 _sampleToCamera, ThinLens _thinLens)
        : sampleToCamera(_sampleToCamera), thinLens(_thinLens) {}
    static Sample2Dto3DMappingSettings sphericalSection(matrix3x3 _sampleToCamera,
                                                        ThinLens _thinLens,
                                                        float _fovXDegrees,
                                                        float _fovYDegrees) {
        Sample2Dto3DMappingSettings result(_sampleToCamera, _thinLens);
        result.type = MappingType::SphericalSection;
        result.fovXDegrees = _fovXDegrees;
        result.fovYDegrees = _fovYDegrees;
        return result;
    }
};

struct SampleHierarchy {
    DynamicArray<RayPacketFrustum3D> tileFrusta3D;
    DynamicArray<RayPacketFrustum3D> blockFrusta3D;
    DynamicArray<DirectionalBeam> directionalSamples;
    void generateFrom2D(const SampleHierarchy2D& hierarchy2D, Sample2Dto3DMappingSettings settings);
};

} // namespace hvvr
