/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "timer.h"

#include <Windows.h>

namespace hvvr {

static double timeScale = 0.0;

class TimerInit {
public:

    TimerInit() {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        timeScale = 1.0 / double(frequency.QuadPart);
    }
};
static TimerInit timerInit;

Timer::Timer() {
    LARGE_INTEGER temp;
    QueryPerformanceCounter(&temp);
    startTime = uint64_t(temp.QuadPart);

    lastTime = startTime;
}

double Timer::get() {
    LARGE_INTEGER temp;
    QueryPerformanceCounter(&temp);
    uint64_t currentTime = uint64_t(temp.QuadPart);

    double rval = (currentTime - lastTime) * timeScale;
    lastTime = currentTime;

    return rval;
}

double Timer::getElapsed() {
    LARGE_INTEGER temp;
    QueryPerformanceCounter(&temp);
    uint64_t currentTime = uint64_t(temp.QuadPart);

    return (currentTime - startTime) * timeScale;
}

} // namespace hvvr
