#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "vector_math.h"

#include <immintrin.h>

struct BVHNode;

namespace hvvr { namespace traverse { namespace avx {

enum { childCount = 4 }; // number of child nodes per BVH entry
enum { childMaskAll = (1 << childCount) - 1 };

// Encoding of a frustum for ray packet traversal
// https://cseweb.ucsd.edu/~ravir/whitted.pdf
// A frustum can be defined by its four corner rays, from which we can derive the planes
// - n[i] = d[i] x d[(i + 1) % 4]
// - b[i] = o[i] . n[i]
// -where n = plane normals, b = plane offsets, o = ray origins, d = ray directions
// -this is specifying n such that the normal points away from the frustum's volume
// -presumably the four corner rays need to be in a particular winding order for this to work
// Fitting a frustum to rays can be done a couple ways:
// -for a pinhole camera, it's simply the corner rays for an NxN group
// --(don't forget to pad these out a bit to fit the volume of the pixels, not just their centers)
// -for the more general case, a two-pass algorithm:
// --find the major axis
// --fit a near and far plane to the major axis, calculate the min/max intersection U,Vs at near/far
// Frustum testing:
// if (n[i] . p[k] - b[i] > 0)
// -where p are the vertices of a polyhedron
// -then p[k] is outside the plane
// -if all p are outside any single plane, then the polyhedron is rejected
//
// We allow for specifying a fully-culling degerenated frustum by using -infinite-b
// (If the b[i] are -inifinity, then the frustum rejects all intersection tests.)
struct Frustum {
    // no near or far plane
    enum { planeCount = 4 };
    enum { pointCount = 4 };

    // primary axis to use for distance estimate?
    // this is an index into the array of floats specifying xMax, xNegMin, etc. in BVHNode
    // -X: -xMax
    // +X: -xNegMin
    // -Y: -yMax
    // +Y: -yNegMin
    // -Z: -zMax
    // +Z: -zNegMin
    int distanceEstimateOffset;
    int pad0[7];

    // precomputed and swizzled for SIMD intersection tests
    __m256 testPlane[planeCount][4];
    __m256 testProjectionX, testProjectionY, testProjectionZ;

    enum { frustumEdgeCount = 4 };
    enum { aabbEdgeCount = 3 };
    enum { edgeCount = frustumEdgeCount * aabbEdgeCount };
    __m256 testEdge[(edgeCount + 7) / 8 * 3];
    __m256 testMinFrustum[(edgeCount + 7) / 8];
    __m256 testMaxFrustum[(edgeCount + 7) / 8];

    // TODO(anankervis): the non-SIMD-swizzled values can probably be removed from
    // non-debug builds to get this structure size down a bit
    vector4 plane[planeCount];
    // these are just here for debugging and external code
    vector3 pointOrigin[pointCount];
    vector3 pointDir[pointCount];

    Frustum() = default;

    // construct from four rays
    Frustum(const vector3* origin, const vector3* dir);

    // intersect against the four child AABBs, corresponding bit index is set if the AABB passes
    // This test is designed to be fast and conservative. Follow with testBVHNodeChildrenRefine
    // to reject additional cases.
    __forceinline uint32_t testBVHNodeChildren(const BVHNode& node) const;
    // intersect against the four child AABBs, corresponding bit index is set if the AABB passes
    // This refines the initial results from testBVHNodeChildren by running additional tests.
    __forceinline uint32_t testBVHNodeChildrenRefine(const BVHNode& node, uint32_t prevResult) const;
};

}}} // namespace hvvr::traverse::avx

#if TRAVERSAL_IMP

#include "avx.h"
#include "bvh_node.h"
#include "constants_math.h"

#include <assert.h>

namespace hvvr { namespace traverse { namespace avx {

__forceinline uint32_t Frustum::testBVHNodeChildren(const BVHNode& node) const {
    enum { childCount = 4 };
    uint32_t result = childMaskAll;

    __m256 aabbX = _mm256_load_ps(node.xMax);
    __m256 aabbY = _mm256_load_ps(node.yMax);
    __m256 aabbZ = _mm256_load_ps(node.zMax);

    // test AABB against frustum faces
    {
        // plane 0
        // n-vertex of AABB for this plane - the most negative along the plane normal
        // n-vertex (and p-vertex) is a permutation of AABB min and max components
        // Compute the component-wise distance for the n-vertex at the same time we select
        // the permutation...
        __m256 nDistSelect =
            _mm256_fmadd_ps(testPlane[0][0], aabbX, _mm256_fmadd_ps(testPlane[0][1], aabbY, testPlane[0][2] * aabbZ));
        // now collapse into a single distance value
        __m128 d = m128(nDistSelect) + extract_m128<1>(nDistSelect);
        // test against plane distance
        __m128 test = (d <= m128(testPlane[0][3]));

        // plane 1
        nDistSelect =
            _mm256_fmadd_ps(testPlane[1][0], aabbX, _mm256_fmadd_ps(testPlane[1][1], aabbY, testPlane[1][2] * aabbZ));
        d = m128(nDistSelect) + extract_m128<1>(nDistSelect);
        test = test & (d <= m128(testPlane[1][3]));

        // plane 2
        nDistSelect =
            _mm256_fmadd_ps(testPlane[2][0], aabbX, _mm256_fmadd_ps(testPlane[2][1], aabbY, testPlane[2][2] * aabbZ));
        d = m128(nDistSelect) + extract_m128<1>(nDistSelect);
        test = test & (d <= m128(testPlane[2][3]));

        // plane 3
        nDistSelect =
            _mm256_fmadd_ps(testPlane[3][0], aabbX, _mm256_fmadd_ps(testPlane[3][1], aabbY, testPlane[3][2] * aabbZ));
        d = m128(nDistSelect) + extract_m128<1>(nDistSelect);
        test = test & (d <= m128(testPlane[3][3]));

        // mask off AABBs which are outside any of the planes
        unsigned int mask = movemask(test);
        result &= mask;
    }

    // everything's already culled - no point in continuing to refine
    if (result == 0)
        return 0;

    // test frustum against AABB faces
    {
        // check for overlap of frustum's projection onto AABB axes
        unsigned int mask =
            movemask((aabbX >= testProjectionX) & (aabbY >= testProjectionY) & (aabbZ >= testProjectionZ));
        mask &= mask >> 4;
        // strip non-overlapping AABBs
        result &= mask;
    }

    return result;
}

__forceinline uint32_t Frustum::testBVHNodeChildrenRefine(const BVHNode& node,
                                                          uint32_t prevResult) const {
    uint32_t result = prevResult;

    // test the remaining axes of separation, the cross products of
    // unique edge direction combinations from the frustum and AABB
    {
        int loadIndex = 0;
        for (int edgeGroup = 0; edgeGroup < (edgeCount + 7) / 8; edgeGroup++) {
            __m256 edgeX = testEdge[loadIndex++];
            __m256 edgeY = testEdge[loadIndex++];
            __m256 edgeZ = testEdge[loadIndex++];

            __m256 pMaskX = edgeX > _mm256_setzero_ps();
            __m256 pMaskY = edgeY > _mm256_setzero_ps();
            __m256 pMaskZ = edgeZ > _mm256_setzero_ps();

            for (int i = 0; i < childCount; i++) {
                // find the p-vertex
                __m256 pX = _mm256_and_ps(pMaskX, _mm256_broadcast_ss(node.xMax + i)) +
                            _mm256_andnot_ps(pMaskX, -_mm256_broadcast_ss(node.xNegMin + i));
                __m256 pY = _mm256_and_ps(pMaskY, _mm256_broadcast_ss(node.yMax + i)) +
                            _mm256_andnot_ps(pMaskY, -_mm256_broadcast_ss(node.yNegMin + i));
                __m256 pZ = _mm256_and_ps(pMaskZ, _mm256_broadcast_ss(node.zMax + i)) +
                            _mm256_andnot_ps(pMaskZ, -_mm256_broadcast_ss(node.zNegMin + i));

                // find the n-vertex
                __m256 nX = _mm256_andnot_ps(pMaskX, _mm256_broadcast_ss(node.xMax + i)) +
                            _mm256_and_ps(pMaskX, -_mm256_broadcast_ss(node.xNegMin + i));
                __m256 nY = _mm256_andnot_ps(pMaskY, _mm256_broadcast_ss(node.yMax + i)) +
                            _mm256_and_ps(pMaskY, -_mm256_broadcast_ss(node.yNegMin + i));
                __m256 nZ = _mm256_andnot_ps(pMaskZ, _mm256_broadcast_ss(node.zMax + i)) +
                            _mm256_and_ps(pMaskZ, -_mm256_broadcast_ss(node.zNegMin + i));

                __m256 maxBox = _mm256_fmadd_ps(edgeX, pX, _mm256_fmadd_ps(edgeY, pY, edgeZ * pZ));
                __m256 minBox = _mm256_fmadd_ps(edgeX, nX, _mm256_fmadd_ps(edgeY, nY, edgeZ * nZ));

                unsigned int mask =
                    movemask((maxBox < testMinFrustum[edgeGroup]) | (testMaxFrustum[edgeGroup] < minBox));
                // mask off any padding lanes
                static_assert(edgeCount < 32, "assumes edgeCount < 32");
                mask &= (1 << (edgeCount - edgeGroup * 8)) - 1;
                if (mask) {
                    // frustum and AABB interval projected onto edge axis do not overlap
                    result &= ~(1 << i);
                }
            }
        }
    }

    return result;
}

// This structure describes the stack frame of the ray caster.
struct BlockFrame {
    struct ALIGN(16) Entry {
        union {
            // cullStack
            struct {
                union {
                    float tDelta; // internal node - size estimate
                    unsigned tDeltaBytes; // leaf node mask (surviving child mask excluding internal nodes)
                };
                float tMinTemp; // distance estimate
            };
            // sortedStack
            struct {
                float tMin; // distance estimate
                // 0 for internal node entries, otherwise the mask of surviving leaf node children
                // a BVHNode's children will create up to 4 stack entries:
                // once for each internal child (mask = 0, node = the child)
                // once for all leaf children (mask = which children are leaves, node = the parent)
                // this mask is used to tell apart internal and leaf node entries
                unsigned mask;
            };
        };
        const BVHNode* node;
    };

    BlockFrame() {
        sortedStackGuardBand->tMin = Infinity;
        cullStackGuardBand->tDelta = -1;
    }
    void sort(uint32_t stackSize);

    // Temporary values produced/consumed during traversal.
    ALIGN(16) float tMin[4];
    ALIGN(16) float tDeltaMem[4];

    enum { stackSize = 64 }; // this is also the limit for how many nodes block cull can emit

    // The block cull stack, produced by the coarse culling phase.
    Entry cullStackGuardBand[1];
    Entry cullStack[stackSize];

    // The sorted tile stack, produced by sorting the cull stack, cloned into the tile stack before each tile is
    // traversed.
    Entry sortedStackGuardBand[1];
    Entry sortedStack[stackSize];
};

struct TileFrame {
    struct ALIGN(16) Entry {
        float tMin;
        uint32_t mask;
        const BVHNode* node;
    };

    TileFrame() {
        stackGuardBand->tMin = Infinity;
    }

    // Temporary values produced/consumed during traversal.
    ALIGN(16) float tMin[4];

    enum { stackSize = 1024 * 4 };

    // The tile stack, consumed during tile traversal.
    Entry stackGuardBand[1];
    Entry stack[stackSize];
};

static_assert(sizeof(BlockFrame::Entry) == sizeof(TileFrame::Entry),
              "Block and tile stack entry structs must have equal size");

// Insertion sorts the culling stack into the sorted stack.
// Assumes that stackSize > 0
inline void BlockFrame::sort(uint32_t stackSize) {
    // After block cull, the cull stack is sorted such that leaf nodes are at the bottom, followed by internal
    // nodes, with the largest internal nodes at the top of the stack (as measured along the primary traversal axis).
    // This function re-sorts from that ordering to ordering by conservative distance along the ray, with the closest
    // nodes at the top of the stack.
    uint32_t top = 0;
    do {
        auto data = shuffle<1, 0, 2, 3>(load_m128(&(cullStack + top)->tDelta));
        auto slot = sortedStack + top;
        __m128 entry;
        goto LOOP_ENTRY;
        do {
            store(&slot->tMin, entry);
            slot--;
        LOOP_ENTRY:
            entry = load_m128(&(slot - 1)->tMin);
        } while (_mm_comilt_ss(entry, data));
        store(&slot->tMin, data);
        if ((cullStack + top)->tDeltaBytes > 0xf)
            slot->mask = 0;
    } while (++top != stackSize);
}

__forceinline uint32_t traverseBlocks(BlockFrame& frame,
                                      const BVHNode* node,
                                      const Frustum& frustum) {
    if (frustum.plane[0].w == -std::numeric_limits<float>::infinity()) {
        return 0;
    }
    auto negateMask = m256(-0.0f);
    uint32_t top = 0;
    auto tMin = 0.0f;
    goto TEST;

POP:
    // up to 4 entries can replace the current entry:
    // 3 internal nodes + 1 set of leaf nodes, or
    // 4 internal nodes
    if (top >= BlockFrame::stackSize - 4 || // Don't flatten too much of the hierarchy before tile cull, that could
                                            // break the advantage of a BVH.
        top == 0 ||                         // We processed everything, and nothing remains
        (frame.cullStack - 1 + top)->tDeltaBytes <= 0xf) { // We processed everything, and only leaf nodes remain
        return top;
    }

    --top;
    node = (frame.cullStack + top)->node;
    tMin = (frame.cullStack + top)->tMinTemp;
TEST:
    // block frustum vs 4x AABB of children, corresponding bit is set if AABB is hit
    unsigned int mask = frustum.testBVHNodeChildren(*node);
    // spend some additional time to refine the test results
    mask = frustum.testBVHNodeChildrenRefine(*node, mask);

    // mask is currently bitmask of which children are hit by ray
    if (!mask)
        goto POP;

    auto leaf = mask & node->boxData.leafMask;
    if (leaf) {
        mask &= ~leaf;

        // insert leaf nodes into the bottom of the stack
        __m128 entry;
        auto i = top;
        goto PUSH_LEAF_LOOP_ENTRY;
        do {
            store(&(frame.cullStack + i)->tDelta, entry);
            --i;
        PUSH_LEAF_LOOP_ENTRY:
            entry = load_m128(&(frame.cullStack - 1 + i)->tDelta);
        } while (_mm_comigt_ss(entry, m128(0)));
        ++top;
        (frame.cullStack + i)->node = node;
        (frame.cullStack + i)->tDeltaBytes = leaf;
        (frame.cullStack + i)->tMinTemp = tMin;
    }

    // mask is currently bitmask of which non-leaf children are hit by ray
    if (!mask)
        goto POP;

    // conservative min ray bundle distance to children along the primary traversal axis (stick it into a temp)
    store(frame.tMin,
          _mm_xor_ps(m128(negateMask), load_m128((float*)((uintptr_t)node + frustum.distanceEstimateOffset))));
    // size of children along the primary traversal axis
    // assumes BVHNode stores float xMax[4] at a multiple of 32 bytes, immediately followed 16 bytes later by xNegMin[4]
    // (and the same for y and z)
    int maxOffset =
        frustum.distanceEstimateOffset & ~16; // remove neg/pos sign choice from distanceEstimateOffset to get Max
    int negMinOffset = maxOffset + 16;        // offset to get NegMin
    store(frame.tDeltaMem,
          load_m128((float*)((uintptr_t)node + maxOffset)) + load_m128((float*)((uintptr_t)node + negMinOffset)));

    // insert each child into the stack, sorting such that the top of the stack is the node
    // with the largest dimensions along the primary traversal axis
    goto CHILD_LOOP_ENTRY;
    do {
        ++top;
    CHILD_LOOP_ENTRY:
        uint32_t k = tzcnt(mask);
        __m128 entry, tTempDelta = _mm_load_ss(frame.tDeltaMem + k);
        auto i = top;
        goto PUSH_LOOP_ENTRY;
        do {
            store(&(frame.cullStack + i)->tDelta, entry);
            --i;
        PUSH_LOOP_ENTRY:
            entry = load_m128(&(frame.cullStack - 1 + i)->tDelta);
        } while (_mm_comigt_ss(entry, tTempDelta));
        (frame.cullStack + i)->node = node + node->boxData.children.offset[k];
        _mm_store_ss(&(frame.cullStack + i)->tDelta, tTempDelta);
        (frame.cullStack + i)->tMinTemp = frame.tMin[k];
        mask = clearLowestBit(mask);
    } while (mask);
    ++top;
    goto POP;
}

__forceinline uint32_t traverseTiles(uint32_t* triIndices,
                                     uint32_t maxTriCount,
                                     TileFrame& frame,
                                     uint32_t top,
                                     const Frustum& frustum) {
    (void)maxTriCount;
    if (frustum.plane[0].w == -std::numeric_limits<float>::infinity()) {
        return 0;
    }
    auto negateMask = m256(-0.0f);
    uint32_t triCount = 0;
    unsigned mask, leaf;
    --top;
    goto POP_LAST;

POP:
    // pop the node stack
    if (!top)
        return triCount;
    --top;
// if we know the stack has at least one entry, and top is already pointing to the next entry, we can skip
// the setup work in POP
POP_LAST:
    // grab a node to test the children of
    auto node = (frame.stack + top)->node;

TEST:
    // tile frustum vs 4x AABB of children, corresponding bit is set if AABB is hit
    mask = frustum.testBVHNodeChildren(*node);

    // no children hit? Move onto the next node
    if (!mask)
        goto POP;

    // were any of the hit children leaf nodes? If so, add the children to the output list.
    leaf = mask & node->boxData.leafMask;
    if (leaf) {
        // spend some additional time to refine the test results
        // this is a CPU vs GPU tradeoff
        // potentially more time culling on the CPU, in exchange for fewer triangles on the GPU
        mask = frustum.testBVHNodeChildrenRefine(*node, mask);

        leaf = mask & node->boxData.leafMask;
        if (leaf) {
            uint32_t emitMask = leaf;
            do {
                auto k = tzcnt(emitMask);
                auto tri = node->boxData.leaf.triIndex[k], e = (node->boxData.leaf.triIndex + 1)[k];
                do {
                    triIndices[triCount++] = tri;

                    // MAX_TRI_INDICES_TO_INTERSECT too small for these parameters, try increasing TILE_SIZE
                    assert(triCount <= maxTriCount);
                } while (++tri != e);
                emitMask = clearLowestBit(emitMask);
            } while (emitMask);
        }
    }

    // strip leaf nodes from the hit mask, now we only care about the remaining hit internal nodes
    mask &= ~node->boxData.leafMask;
    // if there's nothing left, move onto the next stack entry
    if (!mask)
        goto POP;

    // This is needed because block cull inserts nodes that contain a child mix of hit leaf nodes and hit internal
    // nodes up to 4 times into the stack. Leaf node children are grouped into a single entry w/ hit mask, internal node
    // children each get their own entry. This test prevents the leaf entry from spawning duplicate copies of the
    // prepopulated sibling internal nodes.
    // mask will be 0 for internal nodes, or the leaf hit mask for nodes which contain leaf children
    if ((frame.stack + top)->mask)
        goto POP;

    // conservative min ray bundle distance to children along the primary traversal axis (stick it into a temp)
    store(frame.tMin,
          _mm_xor_ps(m128(negateMask), load_m128((float*)((uintptr_t)node + frustum.distanceEstimateOffset))));

    auto k = tzcnt(mask);        // index of first hit child node
                                 // If I only hit one child, avoid stack
    if (!clearLowestBit(mask)) { // only one bit set?
        // if we only hit one child node, avoid inserting into the stack. Directly replace the current node
        // with the hit node, then jump to testing it.
        node = node + node->boxData.children.offset[k];
        goto TEST;
    }
    // otherwise, we hit multiple child nodes and need to insert each of them into the stack, one at a time,
    // sorted by their distance estimates
    goto CHILD_LOOP_ENTRY;
    do {
        ++top;           // make room on the stack
        k = tzcnt(mask); // index of first hit child node

    CHILD_LOOP_ENTRY:
        __m128 tMinTemp = _mm_load_ss(frame.tMin + k); // grab the temp distance estimate we stored earlier
        // search for an insertion point, bubbling up existing entries with smaller distances
        // such that the top of the stack has the element with the least distance
        __m128 entry;
        auto i = top;
        goto PUSH_LOOP_ENTRY;
        do {
            store(&(frame.stack + i)->tMin, entry); // this is a 128-bit store containing a whole stack entry
            --i;
        PUSH_LOOP_ENTRY:
            entry = load_m128(&(frame.stack - 1 + i)->tMin);
        } while (_mm_comilt_ss(entry, tMinTemp));

        // we've found our insertion point, place the child there
        (frame.stack + i)->node = node + node->boxData.children.offset[k];
        (frame.stack + i)->mask = 0;
        _mm_store_ss(&(frame.stack + i)->tMin, tMinTemp);

        mask = clearLowestBit(mask);
    } while (mask);

    goto POP_LAST;
}

}}} // namespace hvvr::traverse::avx

#endif // TRAVERSAL_IMP
