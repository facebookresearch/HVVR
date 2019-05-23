#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bvh_node.h"
#include "vector_math.h"

namespace hvvr { namespace traverse { namespace ref {

enum { childCount = 4 }; // number of child nodes per BVH entry
enum { childMaskAll = (1 << childCount) - 1 };

struct Frustum {
    // no near or far plane
    enum { planeCount = 4 };
    enum { pointCount = 4 };

    // TODO: This doesn't really belong here, it's a hack to enable discarding frusta culled by
    // clipping planes earlier in the pipeline... ideally, those frusta shouldn't make it this far.
    bool degenerate;

    // direction + distance
    vector4 plane[planeCount];

    // primary axis to use for distance estimate
    // this is an index into the array of floats specifying xMax, xNegMin, etc. in BVHNode
    // -X: -xMax
    // +X: -xNegMin
    // -Y: -yMax
    // +Y: -yNegMin
    // -Z: -zMax
    // +Z: -zNegMin
    int distanceEstimateOffset;

    // projection of the frustum onto the X, Y, Z axes
    vector3 projMin;
    vector3 projMax;

    enum { frustumEdgeCount = 4 };
    enum { aabbEdgeCount = 3 };
    enum { edgeCount = frustumEdgeCount * aabbEdgeCount };
    vector3 refineEdge[edgeCount];
    float refineMin[edgeCount];
    float refineMax[edgeCount];

    // these are just here for debugging and external code
    vector3 pointOrigin[pointCount];
    vector3 pointDir[pointCount];

    Frustum() = default;

    // construct from four rays
    Frustum(const vector3* origin, const vector3* dir);

    // intersect against the four child AABBs, corresponding bit index is set if the AABB passes
    // This test is designed to be fast and conservative. Follow with testBVHNodeChildrenRefine
    // to reject additional cases.
    uint32_t testBVHNodeChildren(const BVHNode& node) const;
    // intersect against the four child AABBs, corresponding bit index is set if the AABB passes
    // This refines the initial results from testBVHNodeChildren by running additional tests.
    uint32_t testBVHNodeChildrenRefine(const BVHNode& node, uint32_t prevResult) const;
};

}}} // namespace hvvr::traverse::ref

#if TRAVERSAL_IMP

#include <float.h>

namespace hvvr { namespace traverse { namespace ref {

struct BlockFrame {
    struct Entry {
        union {
            float tDelta; // internal node - size estimate
            uint32_t tDeltaBytes; // leaf node mask (surviving child mask excluding internal nodes)
        };
        float tMinTemp; // distance estimate
        const BVHNode* node;
    };

    BlockFrame() = default;

    enum { stackSize = 64 }; // this is also the limit for how many nodes block cull can emit
    Entry stack[stackSize]; // block cull stack, produced by the coarse culling phase
};

struct TileFrame {
    struct Entry {
        float tMin; // distance estimate
        // 0 for internal node entries, otherwise the mask of surviving leaf node children
        // a BVHNode's children will create up to 4 stack entries:
        // once for each internal child (mask = 0, node = the child)
        // once for all leaf children (mask = which children are leaves, node = the parent)
        // this mask is used to tell apart internal and leaf node entries
        uint32_t mask;
        const BVHNode* node;

        Entry() = default;

        Entry(const BlockFrame::Entry& blockEntry)
            : tMin(blockEntry.tMinTemp)
            , mask(0)
            , node(blockEntry.node) {
            if (blockEntry.tDeltaBytes <= childMaskAll)
                mask = blockEntry.tDeltaBytes;
        }
    };

    TileFrame() = default;

    TileFrame(const BlockFrame& blockFrame, uint32_t stackSize);
    TileFrame(const TileFrame& blockFrame, uint32_t stackSize);

    enum { stackSize = 1024 * 4 };
    Entry stack[stackSize]; // the tile stack, consumed during tile traversal
};

uint32_t traverseBlocks(BlockFrame& frame, const BVHNode* node, const Frustum& frustum);
uint32_t traverseTiles(
    uint32_t* triIndices, uint32_t maxTriCount,
    TileFrame& frame, uint32_t top, const Frustum& frustum);

}}} // namespace hvvr::traverse::ref

#endif // TRAVERSAL_IMP
