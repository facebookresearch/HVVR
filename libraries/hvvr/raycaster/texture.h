#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "raycaster_common.h"

#include <assert.h>

namespace hvvr {

enum class TextureFormat : uint32_t {
    none = 0,
    r8g8b8a8_unorm_srgb,
    r8g8b8a8_unorm,
    r16g16b16a16_unorm,
    r32g32b32a32_float,
    r16g16b16a16_float,
    r11g11b10_float,
    r32_float,

    count
};

struct TextureData {
    const uint8_t* data;
    TextureFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t strideElements;
};

class Texture {
public:
    Texture(const TextureData& textureData);
    ~Texture();

    Texture(const Texture&) = delete;
    Texture(Texture&&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&&) = delete;

    uint32_t getID() const {
        return _textureID;
    }

protected:
    uint32_t _textureID;
};

inline size_t getTextureSize(uint32_t strideElements, uint32_t height, TextureFormat format) {
    size_t elementCount = size_t(strideElements) * height;
    switch (format) {
        case TextureFormat::r8g8b8a8_unorm_srgb:
            return elementCount * 4;
        case TextureFormat::r8g8b8a8_unorm:
            return elementCount * 4;
        case TextureFormat::r16g16b16a16_unorm:
            return elementCount * 8;
        case TextureFormat::r32g32b32a32_float:
            return elementCount * 16;
        case TextureFormat::r16g16b16a16_float:
            return elementCount * 8;
        case TextureFormat::r11g11b10_float:
            return elementCount * 4;
        case TextureFormat::r32_float:
            return elementCount * 4;
    }
    assert(false);
    return 0;
}

// TODO(anankervis): refactor texture creation and destruction
// Returns index of texture in atlas.
uint32_t CreateTexture(const TextureData& textureData);

void DestroyAllTextures();

} // namespace hvvr
