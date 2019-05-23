#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Pulled out all magic numbers into this file. These should eventually be configurable
// but having them in one place and called out is better than littering them everywhere
#include "raycaster_common.h"

#include <stdlib.h>

// Must be large enough to hold maximum # of triIndices (ever)
// If you are mulitthreading, the max per thread is this number divided by number of threads
static const uint32_t MAX_TRI_INDICES_TO_INTERSECT = 1024 * 1024 * 128;
