#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// configuration options for the raycaster

#include <math.h>

enum class RaycasterOutputFormat { COLOR_RGBA8 };

enum class RayCasterGPUMode {
    GPU_INTERSECT_AND_RECONSTRUCT_DEFERRED_MSAA_RESOLVE,
    GPU_FOVEATED_POLAR_SPACE_CUDA_RECONSTRUCT
};

struct RayCasterSpecification {
    RayCasterGPUMode mode = RayCasterGPUMode::GPU_INTERSECT_AND_RECONSTRUCT_DEFERRED_MSAA_RESOLVE;

    // limit the number of threads used by the raycaster when going wide
    // 0 = default = number of hardware threads in the system
    size_t threadCount = 0;

    struct FoveatedSamplePattern {
        float degreeTrackingError = 0.5f;
        float maxFOVDegrees = 110.0f;
        float marSlope = 0.022f;
        float fovealMARDegrees = 1.0f / 60.0f;
    } foveatedSamplePattern;

    RaycasterOutputFormat outputFormat = RaycasterOutputFormat::COLOR_RGBA8;

    static RayCasterSpecification feb2017FoveatedDemoSettings() {
        RayCasterSpecification spec;
        spec.mode = RayCasterGPUMode::GPU_FOVEATED_POLAR_SPACE_CUDA_RECONSTRUCT;
        spec.foveatedSamplePattern.degreeTrackingError = 7.0f;
        spec.foveatedSamplePattern.fovealMARDegrees = 1.0f / 20.0f;
        spec.foveatedSamplePattern.marSlope = 0.015f;
        spec.foveatedSamplePattern.maxFOVDegrees = 90.0f;
        spec.outputFormat = RaycasterOutputFormat::COLOR_RGBA8;
        return spec;
    };

    bool outputTo3DApi = true;
};
