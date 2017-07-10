#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gpu_camera.h"
#include "gpu_scene_state.h"

#include <vector>

namespace hvvr {

// Would use a HashStringMap on names, but VS2013 vs. 2015 troubles. Besides,
// a linear array search is faster at sane camera counts
struct GPUContext {
    std::vector<GPUCamera> cameras;
    GPUSceneState sceneState;
    bool graphicsResourcesMapped = false;

    void getCudaGraphicsResources(std::vector<cudaGraphicsResource_t>& resources);

    // Must be called before accessing any of the dx11/cuda interop textures from cuda
    void maybeMapResources();

    // Must be called once done with the cuda portion of the frame to release the textures and allow them to be blitted
    void unmapResources();

    void cleanup();

    // Returns a reference to the camera with the given name
    // creating it if necessary
    GPUCamera& getCreateCamera(const Camera* cameraPtr, bool& created);
};

extern GPUContext* gGPUContext;

} // namespace hvvr
