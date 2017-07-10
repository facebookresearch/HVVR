#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "util.h"

#include <stdint.h>


// A bounding volume hierarchy node, optimized for traversal.
struct ALIGN(32) BVHNode {
    float xMax[4], xNegMin[4];
    float yMax[4], yNegMin[4];
    float zMax[4], zNegMin[4];

    // A struct representing a 32 byte section of a BVHNode that contains the connectivity information.
    struct ALIGN(32) BoxData {
        unsigned leafMask;
        union {
            struct Children {
                char PADDING[4];
                uint32_t offset[4];
            } children;
            struct Leaf {
                uint32_t triIndex[5];
            } leaf;
        };
        char PADDING2[8];
    } boxData;
};
static_assert(sizeof(BVHNode) == 128, "BVHNode size changed, was this intentional?");
