#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "blockcull.h"
#include "samples.h"
#include "graphics_types.h"

namespace hvvr {

struct SampleHierarchy {
    DynamicArray<RayPacketFrustum2D> tileFrusta2D;
    DynamicArray<RayPacketFrustum2D> blockFrusta2D;
    DynamicArray<RayPacketFrustum3D> tileFrusta3D;
    DynamicArray<RayPacketFrustum3D> blockFrusta3D;

    void generate(ArrayView<SortedSample> sortedSamples,
                  uint32_t blockCount,
                  uint32_t validSampleCount,
                  const FloatRect& cullRect,
                  ArrayView<float> blockedSamplePositions,
                  ArrayView<Sample::Extents> blockedSampleExtents,
                  ThinLens thinLens,
                  const matrix3x3& sampleToCamera);

    void populate3DFrom2D(uint32_t blockCount, const matrix3x3& sampleToCamera, ThinLens thinLens);
};

} // namespace hvvr
