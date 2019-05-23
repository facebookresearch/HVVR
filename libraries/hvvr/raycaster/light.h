#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "vector_math.h"

#include <stdint.h>
#include <vector_types.h>


namespace hvvr {

enum class LightType : uint32_t {
    none = 0,
    directional,
    point,
    spot,

    count
};

struct LightDirectional {
    vector3 Direction;
    vector3 Power;
};

struct LightPoint {
    vector3 Position;
    float FalloffEnd;
    vector3 Color;
    float FalloffScale; // 1 / (end - start)
};

struct LightSpot {
    vector3 Position;
    float FalloffEnd;
    vector3 Direction;
    float FalloffScale; // 1 / (end - start)
    vector3 Color;
    float CosOuterAngle;
    float CosAngleScale; //  1 / (cos_inner_a - cos_outer_a)
};

struct LightUnion {
    LightType type;
    union {
        LightDirectional directional;
        LightPoint point;
        LightSpot spot;
    };

    LightUnion() : type(LightType::none) {}
};

class Light {
public:
    Light(const LightUnion& lightUnion);
    ~Light();

    Light(const Light&) = delete;
    Light(Light&&) = delete;
    Light& operator=(const Light&) = delete;
    Light& operator=(Light&&) = delete;

    const LightUnion& getLightUnion() const {
        return _lightUnion;
    }

protected:
    LightUnion _lightUnion;
};

} // namespace hvvr
