#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "cuda_decl.h"

#include <stdint.h>

namespace hvvr {

// TODO(anankervis): merge this file into another file
struct RaycasterGBufferSubsample {
    uint32_t triIndex;
    uint32_t sampleMask;

    template <uint32_t AARate>
    CUDA_HOST_DEVICE static constexpr uint32_t getSampleMaskAll() {
        return (AARate < 32) ? ((uint32_t(1) << AARate) - 1) : ~uint32_t(0);
    }
};

} // namespace hvvr
