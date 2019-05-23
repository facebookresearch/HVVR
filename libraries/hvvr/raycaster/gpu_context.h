#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gpu_camera.h"
#include "gpu_scene_state.h"

#include <memory>
#include <vector>

namespace hvvr {

// TODO(anankervis): merge with Raycaster class
class GPUContext {
public:
    std::vector<std::unique_ptr<GPUCamera>> cameras;
    GPUSceneState sceneState;
    bool graphicsResourcesMapped;

    // If forceNonTcc is true, select a cuda device with a non-TCC driver.
    static bool cudaInit(bool forceNonTcc);
    static void cudaCleanup();

    GPUContext();
    ~GPUContext();

    void getCudaGraphicsResources(std::vector<cudaGraphicsResource_t>& resources);

    // Must be called before accessing any of the dx11/cuda interop textures from cuda
    void interopMapResources();
    // Must be called once done with the cuda portion of the frame to release the textures and allow them to be blitted
    void interopUnmapResources();

    void cleanup();

    GPUCamera* getCreateCamera(const Camera* cameraPtr, bool& created);

protected:
};

} // namespace hvvr
