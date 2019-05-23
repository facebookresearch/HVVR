#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_decl.h"
#include "vector_math.h"

namespace hvvr {

CHDI float sRgbToLinear(float x) {
    return (x <= 0.04045f) ? x * (1.0f / 12.92f) : powf((x + .055f) * (1.0f / 1.055f), 2.4f);
}
CHDI vector3 sRgbToLinear(const vector3& color) {
    return vector3(sRgbToLinear(color.x), sRgbToLinear(color.y), sRgbToLinear(color.z));
}
CHDI vector4 sRgbToLinear(const vector4& color) {
    return vector4(sRgbToLinear(color.x), sRgbToLinear(color.y), sRgbToLinear(color.z), color.w);
}

CHDI float linearToSRgb(float x) {
    return (x <= 0.0031308f) ? (x * 12.92f) : (powf(x, (1.0f / 2.4f)) * 1.055f - .055f);
}
CHDI vector3 linearToSRgb(const vector3& color) {
    return vector3(linearToSRgb(color.x), linearToSRgb(color.y), linearToSRgb(color.z));
}
CHDI vector4 linearToSRgb(const vector4& color) {
    return vector4(linearToSRgb(color.x), linearToSRgb(color.y), linearToSRgb(color.z), color.w);
}

} // namespace hvvr
