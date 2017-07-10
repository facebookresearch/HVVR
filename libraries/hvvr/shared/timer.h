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

class Timer {
public:

    Timer();

    // time since last call to get()
    double get();

    // time since initialization
    double getElapsed();

private:

    uint64_t startTime;
    uint64_t lastTime;
};

} // namespace hvvr
