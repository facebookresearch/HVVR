#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// configuration options for the raycaster

#include <math.h>

enum class RaycasterOutputMode { COLOR_RGBA8 };

struct FoveatedReconstructionSpecification {
    enum class Mode { BARYCENTRIC, KNN } mode = Mode::BARYCENTRIC;
    size_t computeK = 9;
    // Empirically, computeK 9 or 8 requires triangleSearchK = 15
    // computeK 7 or 6 requires triangleSearch = 14
    // For "high-quality" choose 9,15
    // For (slightly) better perf, use 7,14
    static const size_t triangleSearchK = 15;
};

struct RayCasterSpecification {
    enum class GPUMode {
        GPU_INTERSECT_AND_RECONSTRUCT_DEFERRED_MSAA_RESOLVE,
        GPU_FOVEATED_POLAR_SPACE_CUDA_RECONSTRUCT
    };
    GPUMode mode = GPUMode::GPU_INTERSECT_AND_RECONSTRUCT_DEFERRED_MSAA_RESOLVE;

    // limit the number of threads used by the raycaster when going wide
    // 0 = default = number of hardware threads in the system
    size_t threadCount = 0;

    struct FoveatedSamplePattern {
        float degreeTrackingError = 0.5f;
        float minMAR = 1.0f / 60.0f;
        float maxMAR = INFINITY;
        float maxFOVDegrees = 110.0f;
        float marSlope = 0.022f;
        float fovealMARDegrees = 1.0f / 60.0f;
        float zenithJitterStrength = 0.0f;
        float ringJitterStrength = 0.0f;
    } foveatedSamplePattern;

    FoveatedReconstructionSpecification reconstruction;

    RaycasterOutputMode outputMode = RaycasterOutputMode::COLOR_RGBA8;

    static RayCasterSpecification feb2017FoveatedDemoSettings() {
        RayCasterSpecification spec;
        spec.mode = RayCasterSpecification::GPUMode::GPU_FOVEATED_POLAR_SPACE_CUDA_RECONSTRUCT;
        spec.foveatedSamplePattern.degreeTrackingError = 7.0f;
        spec.foveatedSamplePattern.minMAR = 1.0f / 20.0f;
        spec.foveatedSamplePattern.fovealMARDegrees = 1.0f / 20.0f;
        spec.foveatedSamplePattern.marSlope = 0.015f;
        spec.foveatedSamplePattern.zenithJitterStrength = 0.5f;
        spec.foveatedSamplePattern.ringJitterStrength = 1.0f;
        spec.foveatedSamplePattern.maxFOVDegrees = 90.0f;
        // spec.foveatedSamplePattern.maxMAR = 1.0f / 4.0f; // less than half pixel density of DK1
        spec.outputMode = RaycasterOutputMode::COLOR_RGBA8;
        return spec;
    };
};
