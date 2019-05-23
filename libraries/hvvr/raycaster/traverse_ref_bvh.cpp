#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define TRAVERSAL_IMP 1
#include "traverse_ref.h"

#include <assert.h>

namespace hvvr { namespace traverse { namespace ref {

// Insertion sorts the culling stack into the sorted stack.
// Assumes that stackSize > 0
// After block cull, the cull stack is sorted such that leaf nodes are at the bottom, followed by internal
// nodes, with the largest internal nodes at the top of the stack (as measured along the primary traversal axis).
// This function sorts by conservative distance along the ray, with the closest nodes at the top of the stack.
TileFrame::TileFrame(const BlockFrame& blockFrame, uint32_t stackSize) {
    for (uint32_t top = 0; top < stackSize; top++) {
        TileFrame::Entry e(blockFrame.stack[top]);

        uint32_t slot = top;
        for (; slot > 0; slot--) {
            if (stack[slot - 1].tMin >= e.tMin)
                break;
            stack[slot] = stack[slot - 1];
        }

        stack[slot] = e;
    }
}

TileFrame::TileFrame(const TileFrame& srcFrame, uint32_t stackSize) {
    memcpy(stack, srcFrame.stack, sizeof(Entry) * stackSize);
}

uint32_t traverseBlocks(BlockFrame& frame, const BVHNode* node, const Frustum& frustum) {
    if (frustum.degenerate)
        return 0;

    uint32_t top = 0; // stack size, excluding the node we're currently processing
    float tMin = 0.0f; // distance estimate for the current node
    while (true) {
        // block frustum vs 4x AABB of children, corresponding bit is set if AABB is hit
        uint32_t mask = frustum.testBVHNodeChildren(*node);
        // spend some additional time to refine the test results
        mask = frustum.testBVHNodeChildrenRefine(*node, mask);

        // mask is currently bitmask of which children are hit by the frustum
        if (!mask)
            goto POP;

        uint32_t leaf = mask & node->boxData.leafMask;
        if (leaf) {
            mask &= ~leaf;

            // insert leaf nodes into the bottom of the stack
            uint32_t slot = top++;
            for (; slot > 0; slot--) {
                if (frame.stack[slot - 1].tDelta <= 0) // is the next-lowest slot on the stack a leaf node?
                    break;
                frame.stack[slot] = frame.stack[slot - 1];
            }
            frame.stack[slot].node = node;
            frame.stack[slot].tDeltaBytes = leaf;
            frame.stack[slot].tMinTemp = tMin;
        }

        // mask is currently bitmask of which non-leaf children are hit by ray
        if (!mask)
            goto POP;

        // conservative min ray bundle distance to children along the primary traversal axis
        float childDist[childCount];
        for (int c = 0; c < childCount; c++) {
            childDist[c] = -((float*)((uintptr_t)node + frustum.distanceEstimateOffset))[c];
        }

        // size of children along the primary traversal axis
        // assumes BVHNode stores float xMax[4] at a multiple of 32 bytes, immediately followed 16 bytes later by xNegMin[4]
        // (and the same for y and z)
        int maxOffset =
            frustum.distanceEstimateOffset & ~16; // remove neg/pos sign choice from distanceEstimateOffset to get Max
        int negMinOffset = maxOffset + 16;        // offset to get NegMin
        float childSize[childCount];
        for (int c = 0; c < childCount; c++) {
            childSize[c] = 
                ((float*)((uintptr_t)node + maxOffset))[c] +
                ((float*)((uintptr_t)node + negMinOffset))[c];
        }

        // insert each child into the stack, sorting such that the top of the stack is the node
        // with the largest dimensions along the primary traversal axis
        for (int c = 0; c < childCount; c++) {
            if ((mask & (1 << c)) == 0)
                continue; // this child was culled

            uint32_t slot = top++;
            for (; slot > 0; slot--) {
                if (frame.stack[slot - 1].tDelta <= childSize[c])
                    break;
                frame.stack[slot] = frame.stack[slot - 1];
            }
            frame.stack[slot].node = node + node->boxData.children.offset[c];
            frame.stack[slot].tDelta = childSize[c];
            frame.stack[slot].tMinTemp = childDist[c];
        }

POP:
        // up to 4 entries can replace the current entry:
        // 3 internal nodes + 1 set of leaf nodes, or
        // 4 internal nodes
        if (
            // Don't flatten too much of the hierarchy before tile cull, that could break the advantage of a BVH.
            top >= BlockFrame::stackSize - childCount ||
            // We processed everything, and nothing remains
            top == 0 ||
            // We processed everything, and only leaf nodes remain
            (frame.stack + top - 1)->tDeltaBytes <= childMaskAll) {
            return top;
        }

        top--;
        node = (frame.stack + top)->node;
        tMin = (frame.stack + top)->tMinTemp;
    }
}

uint32_t traverseTiles(
    uint32_t* triIndices, uint32_t maxTriCount,
    TileFrame& frame, uint32_t top, const Frustum& frustum)
{
    (void)maxTriCount;
    if (frustum.degenerate)
        return 0;

    uint32_t triCount = 0;
    while (top > 0) {
        // pop the node stack
        top--;
        // grab a node to test the children of
        const BVHNode* node = frame.stack[top].node;

        // tile frustum vs 4x AABB of children, corresponding bit is set if AABB is hit
        uint32_t mask = frustum.testBVHNodeChildren(*node);

        // no children hit? Move onto the next node
        if (!mask)
            continue;

        // were any of the hit children leaf nodes? If so, add the children to the output list.
        uint32_t leaf = mask & node->boxData.leafMask;
        if (leaf) {
            // spend some additional time to refine the test results
            // this is a tradeoff between extra traversal time, vs extra intersection time/memory
            // if we don't do this, we'll end up with more triangles to test later on
            mask = frustum.testBVHNodeChildrenRefine(*node, mask);

            leaf = mask & node->boxData.leafMask;
            if (leaf) {
                for (int c = 0; c < childCount; c++) {
                    if ((leaf & (1 << c)) == 0)
                        continue; // this child was culled

                    uint32_t triIndexStart = node->boxData.leaf.triIndex[c];
                    uint32_t triIndexEnd = node->boxData.leaf.triIndex[c + 1];
                    for (uint32_t triIndex = triIndexStart; triIndex < triIndexEnd; triIndex++) {
                        assert(triCount < maxTriCount);
                        triIndices[triCount++] = triIndex;
                    }
                }
            }
        }

        // strip leaf nodes from the hit mask, now we only care about the remaining hit internal nodes
        mask &= ~node->boxData.leafMask;
        // if there's nothing left, move onto the next stack entry
        if (!mask)
            continue;

        // This is needed because block cull inserts nodes that contain a child mix of hit leaf nodes and hit internal
        // nodes up to 4 times into the stack. Leaf node children are grouped into a single entry w/ hit mask, internal node
        // children each get their own entry. This test prevents the leaf entry from spawning duplicate copies of the
        // prepopulated sibling internal nodes.
        // mask will be 0 for internal nodes, or the leaf hit mask for nodes which contain leaf children
        if (frame.stack[top].mask)
            continue;

        // conservative min ray bundle distance to children along the primary traversal axis
        float childDist[childCount];
        for (int c = 0; c < childCount; c++) {
            childDist[c] = -((float*)((uintptr_t)node + frustum.distanceEstimateOffset))[c];
        }

        // insert each surviving internal child into the stack, closest node goes at top of stack
        for (int c = 0; c < childCount; c++) {
            if ((mask & (1 << c)) == 0)
                continue; // this child was culled

            uint32_t slot = top++;
            for (; slot > 0; slot--) {
                if (frame.stack[slot - 1].tMin >= childDist[c])
                    break;
                frame.stack[slot] = frame.stack[slot - 1];
            }
            frame.stack[slot].node = node + node->boxData.children.offset[c];
            frame.stack[slot].tMin = childDist[c];
            frame.stack[slot].mask = 0;
        }
    }
    return triCount;
}

}}} // namespace hvvr::traverse::ref
