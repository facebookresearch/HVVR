#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gpu_buffer.h"
#include "gpu_camera.h"
#include "gpu_samples.h"
#include "vector_math.h"


namespace hvvr {

// For conveniently passing four planes by value to the world space transformation kernel.
struct FourPlanes {
    Plane data[4];
};

void ComputeEyeSpaceFrusta(const GPUBuffer<DirectionalBeam>& dirSamples,
                           GPUBuffer<SimpleRayFrustum>& tileFrusta,
                           GPUBuffer<SimpleRayFrustum>& blockFrusta);

void CalculateWorldSpaceFrusta(SimpleRayFrustum* blockFrustaWS,
                               SimpleRayFrustum* tileFrustaWS,
                               SimpleRayFrustum* blockFrustaES,
                               SimpleRayFrustum* tileFrustaES,
                               matrix4x4 eyeToWorldMatrix,
                               FourPlanes cullPlanes,
                               uint32_t blockCount,
                               uint32_t tileCount,
                               cudaStream_t stream);

} // namespace hvvr
