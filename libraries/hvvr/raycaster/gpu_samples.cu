/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gpu_camera.h"
#include "gpu_samples.h"


namespace hvvr {

CameraBeams::CameraBeams(const GPUCamera& camera) {
    directionalBeams = camera.d_batchSpaceBeams;
    frameJitter = camera.frameJitter;
    lens = camera.lens;
}

} // namespace hvvr
