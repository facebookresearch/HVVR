/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "light.h"

#include <assert.h>


namespace hvvr {

Light::Light(const LightUnion& lightUnion) : _lightUnion(lightUnion) {}

Light::~Light() {}

} // namespace hvvr
