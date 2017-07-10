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


struct matrix4x4;

namespace hvvr {

struct GPUCamera;
struct RaycasterGBufferSubsample;

void DeferredIntersectTiles(GPUCamera& camera,
                            SampleInfo sampleInfo,
                            const matrix3x3& sampleToCamera,
                            const matrix4x4& cameraToWorld);


void GenerateExplicitRayBuffer(GPUCamera& camera, SimpleRay* d_rayBuffer);

} // namespace hvvr
