#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_util.h"
#include "texture.h"

namespace hvvr {

// Data structure for 2D texture
struct Texture2D {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t elementSize = 0;
    cudaArray* d_rawMemory = 0;
    cudaMipmappedArray* d_rawMipMappedMemory = 0;
    cudaTextureObject_t d_texObject = 0;
    cudaSurfaceObject_t d_surfaceObject = 0;
    TextureFormat format = TextureFormat::none;
    bool hasMipMaps = false;
    bool hasSurfaceObject = false;
};

Texture2D createEmptyTexture(uint32_t width,
                             uint32_t height,
                             TextureFormat format,
                             cudaTextureAddressMode xWrapMode,
                             cudaTextureAddressMode yWrapMode,
                             bool linearFilter = true);

void clearTexture(Texture2D tex);

} // namespace hvvr
