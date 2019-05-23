#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "light.h"
#include "raycaster_common.h"

namespace hvvr {

struct LightingEnvironment {
    LightDirectional directionalLights[MAX_DIRECTIONAL_LIGHTS];
    LightPoint pointLights[MAX_POINT_LIGHTS];
    LightSpot spotLights[MAX_SPOT_LIGHTS];
    int directionalLightCount;
    int pointLightCount;
    int spotLightCount;
};

} // namespace hvvr
