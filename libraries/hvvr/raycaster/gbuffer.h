#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <stdint.h>

namespace hvvr {

// TODO(anankervis): merge this file into another file
struct RaycasterGBufferSubsample {
    uint32_t triIndex;
    uint32_t sampleMask;
};

} // namespace hvvr
