/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gpu_camera.h"
#include "gpu_samples.h"


namespace hvvr {

SampleInfo::SampleInfo(const GPUCamera& camera) {
    centers = camera.d_sampleLocations;
    extents = camera.d_sampleExtents;
    frameJitter = camera.frameJitter;
    lens = camera.lens;
}

} // namespace hvvr
