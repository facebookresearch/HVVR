#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gpu_samples.h"
#include "graphics_types.h"
#include "samples.h"
#include "traversal.h"

namespace hvvr {

// Encoding of a frustum for ray packet traversal
struct RayPacketFrustum2D {
    // xMin, xNegMax, yMin, yNegMax
    vector4 data;

    RayPacketFrustum2D(float xMin, float xMax, float yMin, float yMax) : data(xMin, -xMax, yMin, -yMax) {}
    RayPacketFrustum2D() = default;

    // Set the mins to infinity and maxs to -infinity
    void setEmpty() {
        data = vector4(std::numeric_limits<float>::infinity());
    }
    void merge(float x, float y) {
        data = min(data, vector4(x, -x, y, -y));
    }
    void merge(const RayPacketFrustum2D& other) {
        data = min(data, other.data);
    }
    void intersect(const RayPacketFrustum2D& other) {
        data = max(data, other.data);
    }

    inline float xMin() const {
        return data.x;
    }
    inline float xMax() const {
        return -data.y;
    }
    inline float yMin() const {
        return data.z;
    }
    inline float yMax() const {
        return -data.w;
    }
    inline float xNegMax() const {
        return data.y;
    }
    inline float yNegMax() const {
        return data.w;
    }
};

struct BeamBatch2D {
    DynamicArray<RayPacketFrustum2D> tileFrusta;
    DynamicArray<RayPacketFrustum2D> blockFrusta;
    DynamicArray<UnpackedSample> samples;
    BeamBatch2D() {}
    BeamBatch2D(ArrayView<SortedSample> sortedSamples,
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

struct BeamBatch {
    DynamicArray<Frustum> tileFrusta3D;
    DynamicArray<Frustum> blockFrusta3D;
    DynamicArray<DirectionalBeam> directionalBeams;
    void generateFrom2D(const BeamBatch2D& hierarchy2D, Sample2Dto3DMappingSettings settings);
};

} // namespace hvvr
