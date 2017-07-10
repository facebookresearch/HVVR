/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "raycaster.h"
#include "camera.h"
#include "cuda_raycaster.h"
#include "light.h"
#include "model.h"
#include "texture.h"
#include "thread_pool.h"


namespace hvvr {

// set FTZ and DAZ to prevent (slow) denormals
static void setFloatMode() {
    _mm_setcsr(_mm_getcsr() | 0x8040);
}

static void threadInit(size_t threadIndex) {
    (void)threadIndex;
    setFloatMode();
}

Raycaster::Raycaster(const RayCasterSpecification& spec) : _spec(spec), _sceneDirty(false) {
    setFloatMode();

    size_t numThreads = max<size_t>(1, std::thread::hardware_concurrency());
    if (_spec.threadCount > 0) {
        numThreads = min(_spec.threadCount, numThreads);
    }
    _threadPool = std::make_unique<ThreadPool>(numThreads, threadInit);

    if (!Init()) {
        assert(false);
    }
}

Raycaster::~Raycaster() {
    _cameras.clear();

    cleanupScene();

    Cleanup();
}

Camera* Raycaster::createCamera(const FloatRect& viewport, float apertureRadius) {
    _cameras.emplace_back(std::make_unique<Camera>(viewport, apertureRadius));
    return (_cameras.end() - 1)->get();
}

void Raycaster::destroyCamera(Camera* camera) {
    for (auto it = _cameras.begin(); it != _cameras.end(); ++it) {
        if (it->get() == camera) {
            _cameras.erase(it);
            return;
        }
    }
    assert(false); // not found
}

} // namespace hvvr
