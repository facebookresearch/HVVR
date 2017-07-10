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

void ComputeEyeSpaceFrusta(const GPUBuffer<PrecomputedDirectionSample>& dirSamples,
                           GPUBuffer<SimpleRayFrustum>& tileFrusta,
                           GPUBuffer<SimpleRayFrustum>& blockFrusta);

void ResetCullFrusta(GPURayPacketFrustum* d_blockFrusta,
                     GPURayPacketFrustum* d_tileFrusta,
                     const uint32_t tileCount,
                     const uint32_t blockCount,
                     cudaStream_t stream);

void CalculateSampleCullFrusta(GPURayPacketFrustum* d_blockFrusta,
                               GPURayPacketFrustum* d_tileFrusta,
                               SampleInfo sampleInfo,
                               const uint32_t sampleCount,
                               const uint32_t tileCount,
                               const uint32_t blockCount,
                               cudaStream_t stream);

void CalculateWorldSpaceFrusta(SimpleRayFrustum* blockFrustaWS,
                               SimpleRayFrustum* tileFrustaWS,
                               SimpleRayFrustum* blockFrustaES,
                               SimpleRayFrustum* tileFrustaES,
                               matrix4x4 eyeToWorldMatrix,
                               uint32_t blockCount,
                               uint32_t tileCount,
                               cudaStream_t stream);

} // namespace hvvr
