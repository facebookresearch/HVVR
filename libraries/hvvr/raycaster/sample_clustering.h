#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dynamic_array.h"
#include "magic_constants.h"
#include "samples.h"
#include <algorithm>

namespace hvvr {

template <class PointClass>
void naiveXYCluster(ArrayView<PointClass> samples, size_t k) {
    size_t yStrata = (size_t)sqrt(k);
    size_t xStrata = (k + yStrata - 1) / yStrata;

    size_t tilesPerBlock = TILES_PER_BLOCK;
    size_t blockYStrata = (size_t)sqrt(tilesPerBlock);
    size_t blockXStrata = (tilesPerBlock + blockYStrata - 1) / blockYStrata;

    auto yThenXComparator = [](const PointClass& a, const PointClass& b) {
        return (a.position.y < b.position.y) || (a.position.y == b.position.y && a.position.x < b.position.x);
    };
    auto xThenYComparator = [](const PointClass& a, const PointClass& b) {
        return (a.position.x < b.position.x) || (a.position.x == b.position.x && a.position.y < b.position.y);
    };

    std::sort(samples.begin(), samples.end(), yThenXComparator);

    for (size_t i = 0; i < yStrata; ++i) {
        std::sort(samples.begin() + (i * BLOCK_SIZE * xStrata),
                  samples.begin() + std::min(((i + 1) * BLOCK_SIZE * xStrata), samples.size()), xThenYComparator);
    }

    for (size_t i = 0; i < k; ++i) {
        std::sort(samples.begin() + (i * BLOCK_SIZE),
                  samples.begin() + std::min(((i + 1) * BLOCK_SIZE), samples.size()), yThenXComparator);
    }

    for (size_t i = 0; i < samples.size(); i += TILE_SIZE * blockXStrata) {
        std::sort(samples.begin() + i, samples.begin() + std::min(i + TILE_SIZE * blockXStrata, samples.size()),
                  xThenYComparator);
    }
}

} // namespace hvvr
