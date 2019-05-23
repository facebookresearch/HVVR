#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "vector_math.h"

namespace hvvr {

enum class ShadingModel : uint32_t {
    none = 0,
    phong,
    emissive,
};

// 64 bytes of material information
struct SimpleMaterial {
    enum : uint32_t { textureIndexBitCount = 16 };
    enum : uint32_t { maxTextureCount = (1 << textureIndexBitCount) };
    enum : uint32_t { textureMask = maxTextureCount - 1 };
    enum : uint32_t { badTextureIndex = textureMask };

    enum : uint32_t { diffuseBitShift = textureIndexBitCount * 3 };
    enum : uint32_t { specularBitShift = textureIndexBitCount * 2 };
    enum : uint32_t { glossyBitShift = textureIndexBitCount };

    static uint64_t buildShadingCode(ShadingModel shadingModel,
                                     uint32_t diffuseOrEmissiveID,
                                     uint32_t specularID,
                                     uint32_t glossinessID) {
        return uint64_t(shadingModel) | uint64_t(diffuseOrEmissiveID) << hvvr::SimpleMaterial::diffuseBitShift |
               uint64_t(glossinessID) << hvvr::SimpleMaterial::glossyBitShift |
               uint64_t(specularID) << hvvr::SimpleMaterial::specularBitShift;
    }

    vector4 emissive;
    vector4 diffuse;
    vector4 specular;
    float glossiness;
    float opacity;
    // upper 16 bits diffuse/emissive | next 16 bits specularTexture | next 16 bits glossinessTexture | lower 16 bits
    // shadingModel
    uint64_t textureIDsAndShadingModel;
};

} // namespace hvvr
