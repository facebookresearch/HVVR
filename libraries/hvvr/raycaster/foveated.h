#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "raycaster_spec.h"
#include "sample_hierarchy.h"
#include "graphics_types.h"

namespace hvvr {

class Raycaster;

// precomputed eye-space derivatives of the direction sample
struct PrecomputedDirectionSample {
    vector3 center; // original ray
    vector3 d1;     // differential ray 1
    vector3 d2;     // differential ray 2
};

struct FoveatedSampleData {
	uint32_t triangleCount = 0;
	size_t validSampleCount = 0;
	size_t blockCount = 0;
	SampleHierarchy samples;
	DynamicArray<SimpleRayFrustum> simpleTileFrusta;
	DynamicArray<SimpleRayFrustum> simpleBlockFrusta;
	DynamicArray<DirectionSample> eyeSpaceSamples;
	DynamicArray<PrecomputedDirectionSample> precomputedEyeSpaceSamples;
};

// TODO(anankervis): merge into Raycaster class
void generateEyeSpacePolarFoveatedSampleData(FoveatedSampleData& foveatedSampleData,
                                             std::vector<vector2ui>& polarRemapToPixel,
                                             std::vector<float>& ringEccentricities,
                                             std::vector<float>& eccentricityRemap,
                                             size_t& samplesPerRing,
                                             RayCasterSpecification::FoveatedSamplePattern pattern);

void polarSpaceFoveatedSetup(Raycaster* raycaster);

} // namespace hvvr
