/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "blockcull.h"
#include "avx.h"
#include "camera.h"
#include "constants_math.h"
#include "gpu_camera.h"
#include "magic_constants.h"
#include "raycaster.h"
#include "thread_pool.h"
#include "timer.h"

#include <stdio.h>

#define DEBUG_STATS 0
#define TIME_BLOCK_CULL 0

#pragma warning(disable : 4505) // unreferenced local function

namespace hvvr {

// TODO(anankervis): split this into a separate file
// TODO(anankervis): optimize this - it gets called a lot
// find the major axis, and precompute some values for the intersection tests
void RayPacketFrustum3D::updatePrecomputed() {
    plane[0] = vector4(cross(pointDir[0], pointDir[1]), 0.0f);
    plane[1] = vector4(cross(pointDir[1], pointDir[2]), 0.0f);
    plane[2] = vector4(cross(pointDir[2], pointDir[3]), 0.0f);
    plane[3] = vector4(cross(pointDir[3], pointDir[0]), 0.0f);
    plane[0].w = dot(pointOrigin[0], vector3(plane[0]));
    plane[1].w = dot(pointOrigin[1], vector3(plane[1]));
    plane[2].w = dot(pointOrigin[2], vector3(plane[2]));
    plane[3].w = dot(pointOrigin[3], vector3(plane[3]));

    vector3 avgDir = pointDir[0] + pointDir[1] + pointDir[2] + pointDir[3];
    if (fabsf(avgDir.x) > fabsf(avgDir.y)) {
        if (fabsf(avgDir.x) > fabsf(avgDir.z)) {
            distanceEstimateOffset = avgDir.x < 0 ? offsetof(BVHNode, xMax) : offsetof(BVHNode, xNegMin);
        } else {
            distanceEstimateOffset = avgDir.z < 0 ? offsetof(BVHNode, zMax) : offsetof(BVHNode, zNegMin);
        }
    } else {
        if (fabsf(avgDir.y) > fabsf(avgDir.z)) {
            distanceEstimateOffset = avgDir.y < 0 ? offsetof(BVHNode, yMax) : offsetof(BVHNode, yNegMin);
        } else {
            distanceEstimateOffset = avgDir.z < 0 ? offsetof(BVHNode, zMax) : offsetof(BVHNode, zNegMin);
        }
    }

    // precomputation for the frustum planes vs AABB test
    {
        for (int planeIndex = 0; planeIndex < planeCount; planeIndex++) {
            // selectors to build n-vertex from AABB max/min
            testPlane[planeIndex][0] = m256(plane[planeIndex].x < 0.0f ? plane[planeIndex].x : 0.0f,
                                            plane[planeIndex].x < 0.0f ? plane[planeIndex].x : 0.0f,
                                            plane[planeIndex].x < 0.0f ? plane[planeIndex].x : 0.0f,
                                            plane[planeIndex].x < 0.0f ? plane[planeIndex].x : 0.0f,
                                            // store negated value to compensate for AABB negated min values
                                            plane[planeIndex].x > 0.0f ? -plane[planeIndex].x : 0.0f,
                                            plane[planeIndex].x > 0.0f ? -plane[planeIndex].x : 0.0f,
                                            plane[planeIndex].x > 0.0f ? -plane[planeIndex].x : 0.0f,
                                            plane[planeIndex].x > 0.0f ? -plane[planeIndex].x : 0.0f);
            testPlane[planeIndex][1] = m256(plane[planeIndex].y < 0.0f ? plane[planeIndex].y : 0.0f,
                                            plane[planeIndex].y < 0.0f ? plane[planeIndex].y : 0.0f,
                                            plane[planeIndex].y < 0.0f ? plane[planeIndex].y : 0.0f,
                                            plane[planeIndex].y < 0.0f ? plane[planeIndex].y : 0.0f,
                                            // store negated value to compensate for AABB negated min values
                                            plane[planeIndex].y > 0.0f ? -plane[planeIndex].y : 0.0f,
                                            plane[planeIndex].y > 0.0f ? -plane[planeIndex].y : 0.0f,
                                            plane[planeIndex].y > 0.0f ? -plane[planeIndex].y : 0.0f,
                                            plane[planeIndex].y > 0.0f ? -plane[planeIndex].y : 0.0f);
            testPlane[planeIndex][2] = m256(plane[planeIndex].z < 0.0f ? plane[planeIndex].z : 0.0f,
                                            plane[planeIndex].z < 0.0f ? plane[planeIndex].z : 0.0f,
                                            plane[planeIndex].z < 0.0f ? plane[planeIndex].z : 0.0f,
                                            plane[planeIndex].z < 0.0f ? plane[planeIndex].z : 0.0f,
                                            // store negated value to compensate for AABB negated min values
                                            plane[planeIndex].z > 0.0f ? -plane[planeIndex].z : 0.0f,
                                            plane[planeIndex].z > 0.0f ? -plane[planeIndex].z : 0.0f,
                                            plane[planeIndex].z > 0.0f ? -plane[planeIndex].z : 0.0f,
                                            plane[planeIndex].z > 0.0f ? -plane[planeIndex].z : 0.0f);

            // plane distance
            testPlane[planeIndex][3] = m256(plane[planeIndex].w, plane[planeIndex].w, plane[planeIndex].w,
                                            plane[planeIndex].w, 0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // precomputation for the AABB planes vs frustum test
    {
        // prep for computing max in the lower 4 floats, -min in the upper 4
        // baking in a hi/lo 128-bit shuffle (a negate)
        __m256 swizzledPointOriginX = -m256(pointOrigin[0].x, pointOrigin[1].x, pointOrigin[2].x, pointOrigin[3].x,
                                            -pointOrigin[0].x, -pointOrigin[1].x, -pointOrigin[2].x, -pointOrigin[3].x);
        __m256 swizzledPointOriginY = -m256(pointOrigin[0].y, pointOrigin[1].y, pointOrigin[2].y, pointOrigin[3].y,
                                            -pointOrigin[0].y, -pointOrigin[1].y, -pointOrigin[2].y, -pointOrigin[3].y);
        __m256 swizzledPointOriginZ = -m256(pointOrigin[0].z, pointOrigin[1].z, pointOrigin[2].z, pointOrigin[3].z,
                                            -pointOrigin[0].z, -pointOrigin[1].z, -pointOrigin[2].z, -pointOrigin[3].z);

        // prep for computing max in the lower 4 floats, -min in the upper 4
        // baking in a hi/lo 128-bit shuffle (a negate)
        // also baking in a max with zero, and scaling by FLT_MAX
        // This will be used to extend the frustum's projection to infinity in the direction
        // the frustum is pointing (the frustum has no far plane).
        __m256 swizzledPointDirX =
            m256(FLT_MAX) *
            _mm256_max_ps(_mm256_setzero_ps(), -m256(pointDir[0].x, pointDir[1].x, pointDir[2].x, pointDir[3].x,
                                                     -pointDir[0].x, -pointDir[1].x, -pointDir[2].x, -pointDir[3].x));
        __m256 swizzledPointDirY =
            m256(FLT_MAX) *
            _mm256_max_ps(_mm256_setzero_ps(), -m256(pointDir[0].y, pointDir[1].y, pointDir[2].y, pointDir[3].y,
                                                     -pointDir[0].y, -pointDir[1].y, -pointDir[2].y, -pointDir[3].y));
        __m256 swizzledPointDirZ =
            m256(FLT_MAX) *
            _mm256_max_ps(_mm256_setzero_ps(), -m256(pointDir[0].z, pointDir[1].z, pointDir[2].z, pointDir[3].z,
                                                     -pointDir[0].z, -pointDir[1].z, -pointDir[2].z, -pointDir[3].z));

        // project the frustum onto the X, Y, Z axes
        // The frustum's four origin points will form one end of the projection, and the projection
        // will extend to infinity on the other end, depending on the sign of the four
        // edge directions. It is possible for a wide frustum to extend to infinity on
        // both ends of a projection.
        testProjectionX = swizzledPointOriginX + swizzledPointDirX;
        testProjectionY = swizzledPointOriginY + swizzledPointDirY;
        testProjectionZ = swizzledPointOriginZ + swizzledPointDirZ;

        // compute the max and -min across all four lanes and replicate
        testProjectionX = max(testProjectionX, shuffle<2, 3, 0, 1>(testProjectionX));
        testProjectionX = max(testProjectionX, shuffle<1, 0, 3, 2>(testProjectionX));

        testProjectionY = max(testProjectionY, shuffle<2, 3, 0, 1>(testProjectionY));
        testProjectionY = max(testProjectionY, shuffle<1, 0, 3, 2>(testProjectionY));

        testProjectionZ = max(testProjectionZ, shuffle<2, 3, 0, 1>(testProjectionZ));
        testProjectionZ = max(testProjectionZ, shuffle<1, 0, 3, 2>(testProjectionZ));

        // the comparison which checks for overlap of the AABB and frustum projections
        // expects the frustum projection to be negated, and for the max/negMin
        // components to be swapped (such that -min comes first, max comes second - the
        // hi/lo 128-bit shuffle above)
        // The result is min, -max for the frustum values and max, -min for the AABB.
        testProjectionX = -testProjectionX;
        testProjectionY = -testProjectionY;
        testProjectionZ = -testProjectionZ;
    }

    // precomputation for the AABB x frustum edge tests
    {
        vector3 edge[(edgeCount + 7) / 8 * 8] = {
            cross(pointDir[0], vector3(1, 0, 0)), cross(pointDir[0], vector3(0, 1, 0)),
            cross(pointDir[0], vector3(0, 0, 1)),

            cross(pointDir[1], vector3(1, 0, 0)), cross(pointDir[1], vector3(0, 1, 0)),
            cross(pointDir[1], vector3(0, 0, 1)),

            cross(pointDir[2], vector3(1, 0, 0)), cross(pointDir[2], vector3(0, 1, 0)),
            cross(pointDir[2], vector3(0, 0, 1)),

            cross(pointDir[3], vector3(1, 0, 0)), cross(pointDir[3], vector3(0, 1, 0)),
            cross(pointDir[3], vector3(0, 0, 1)),
        };

        for (int i = 0; i < (edgeCount + 7) / 8; i++) {
            testEdge[i * 3 + 0] = m256(edge[i * 8 + 0].x, edge[i * 8 + 1].x, edge[i * 8 + 2].x, edge[i * 8 + 3].x,
                                       edge[i * 8 + 4].x, edge[i * 8 + 5].x, edge[i * 8 + 6].x, edge[i * 8 + 7].x);
            testEdge[i * 3 + 1] = m256(edge[i * 8 + 0].y, edge[i * 8 + 1].y, edge[i * 8 + 2].y, edge[i * 8 + 3].y,
                                       edge[i * 8 + 4].y, edge[i * 8 + 5].y, edge[i * 8 + 6].y, edge[i * 8 + 7].y);
            testEdge[i * 3 + 2] = m256(edge[i * 8 + 0].z, edge[i * 8 + 1].z, edge[i * 8 + 2].z, edge[i * 8 + 3].z,
                                       edge[i * 8 + 4].z, edge[i * 8 + 5].z, edge[i * 8 + 6].z, edge[i * 8 + 7].z);
        }

        int loadIndex = 0;
        for (int edgeGroup = 0; edgeGroup < (edgeCount + 7) / 8; edgeGroup++) {
            __m256 edgeX = testEdge[loadIndex++];
            __m256 edgeY = testEdge[loadIndex++];
            __m256 edgeZ = testEdge[loadIndex++];

            __m256 minFrustum = _mm256_set1_ps(FLT_MAX);
            __m256 maxFrustum = _mm256_set1_ps(-FLT_MAX);
            for (int pointIndex = 0; pointIndex < pointCount; pointIndex++) {
                __m256 d0 =
                    _mm256_fmadd_ps(edgeX, _mm256_broadcast_ss((float*)&pointOrigin[pointIndex].x),
                                    _mm256_fmadd_ps(edgeY, _mm256_broadcast_ss((float*)&pointOrigin[pointIndex].y),
                                                    edgeZ * _mm256_broadcast_ss((float*)&pointOrigin[pointIndex].z)));
                minFrustum = min(minFrustum, d0);
                maxFrustum = max(maxFrustum, d0);

                // frustum has no far plane, it extends to infinity
                __m256 d1 =
                    _mm256_fmadd_ps(edgeX, _mm256_broadcast_ss((float*)&pointDir[pointIndex].x),
                                    _mm256_fmadd_ps(edgeY, _mm256_broadcast_ss((float*)&pointDir[pointIndex].y),
                                                    edgeZ * _mm256_broadcast_ss((float*)&pointDir[pointIndex].z)));

                // fmadd has a bit too much precision, it seems... let's not project the
                // frustum to infinity over something that would normally be a rounding error
                __m256 epsilon = _mm256_set1_ps(.000001f);

                minFrustum = _mm256_min_ps(d1 + epsilon, _mm256_setzero_ps()) * _mm256_set1_ps(FLT_MAX) + minFrustum;
                maxFrustum = _mm256_max_ps(d1 - epsilon, _mm256_setzero_ps()) * _mm256_set1_ps(FLT_MAX) + maxFrustum;
            }

            testMinFrustum[edgeGroup] = minFrustum;
            testMaxFrustum[edgeGroup] = maxFrustum;
        }
    }
}

__forceinline unsigned int RayPacketFrustum3D::testBVHNodeChildren(const BVHNode& node) const {
    enum { aabbCount = 4 };
    unsigned int result = (1 << aabbCount) - 1;

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

__forceinline unsigned int RayPacketFrustum3D::testBVHNodeChildrenRefine(const BVHNode& node,
                                                                         unsigned int result) const {
    enum { aabbCount = 4 };

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

            for (int i = 0; i < aabbCount; i++) {
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
struct StackFrameBlock {
    struct ALIGN(16) Entry {
        union {
            struct {
                union {
                    float tDelta;
                    unsigned tDeltaBytes;
                };
                float tMinTemp;
            };
            struct {
                float tMin;
                unsigned mask;
            };
        };
        const BVHNode* node;
    };

    StackFrameBlock() {
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

struct StackFrameTile {
    struct ALIGN(16) Entry {
        float tMin;
        uint32_t mask;
        const BVHNode* node;
    };

    StackFrameTile() {
        stackGuardBand->tMin = Infinity;
    }

    // Temporary values produced/consumed during traversal.
    ALIGN(16) float tMin[4];

    enum { stackSize = 1024 * 4 };

    // The tile stack, consumed during tile traversal.
    Entry stackGuardBand[1];
    Entry stack[stackSize];
};

static_assert(sizeof(StackFrameBlock::Entry) == sizeof(StackFrameTile::Entry),
              "Block and tile stack entry structs must have equal size");

// Insertion sorts the culling stack into the sorted stack.
// Assumes that stackSize > 0
void StackFrameBlock::sort(uint32_t stackSize) {
    // After block cull, the cull stack is sorted such that leaf nodes are at the bottom, followed by internal
    // nodes, with the largest internal nodes at the top of the stack (as measured along the primary traversal axis).
    // This function sorts by conservative distance along the ray, with the closest nodes at the top of the stack.
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

static __forceinline uint32_t blockCull3D(StackFrameBlock* frame,
                                          const BVHNode* node,
                                          const RayPacketFrustum3D& frustum) {
    auto negateMask = m256(-0.0f);
    uint32_t top = 0;
    auto tMin = 0.0f;
    goto TEST;

POP:
    // up to 4 entries can replace the current entry:
    // 3 internal nodes + 1 set of leaf nodes, or
    // 4 internal nodes
    if (top >= StackFrameBlock::stackSize - 4 || // Don't flatten too much of the hierarchy before tile cull, that could
                                                 // break the advantage of a BVH.
        top == 0 ||                              // We processed everything, and nothing remains
        (frame->cullStack - 1 + top)->tDeltaBytes <= 0xf) { // We processed everything, and only leaf nodes remain
        return top;
    }

    --top;
    node = (frame->cullStack + top)->node;
    tMin = (frame->cullStack + top)->tMinTemp;
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
            store(&(frame->cullStack + i)->tDelta, entry);
            --i;
        PUSH_LEAF_LOOP_ENTRY:
            entry = load_m128(&(frame->cullStack - 1 + i)->tDelta);
        } while (_mm_comigt_ss(entry, m128(0)));
        ++top;
        (frame->cullStack + i)->node = node;
        (frame->cullStack + i)->tDeltaBytes = leaf;
        (frame->cullStack + i)->tMinTemp = tMin;
    }

    // mask is currently bitmask of which non-leaf children are hit by ray
    if (!mask)
        goto POP;

    // conservative min ray bundle distance to children along the primary traversal axis (stick it into a temp)
    store(frame->tMin,
          _mm_xor_ps(m128(negateMask), load_m128((float*)((uintptr_t)node + frustum.distanceEstimateOffset))));
    // size of children along the primary traversal axis
    // assumes BVHNode stores float xMax[4] at a multiple of 32 bytes, immediately followed 16 bytes later by xNegMin[4]
    // (and the same for y and z)
    int maxOffset =
        frustum.distanceEstimateOffset & ~16; // remove neg/pos sign choice from distanceEstimateOffset to get Max
    int negMinOffset = maxOffset + 16;        // offset to get NegMin
    store(frame->tDeltaMem,
          load_m128((float*)((uintptr_t)node + maxOffset)) + load_m128((float*)((uintptr_t)node + negMinOffset)));

    // insert each child into the stack, sorting such that the top of the stack is the node
    // with the largest dimensions along the primary traversal axis
    goto CHILD_LOOP_ENTRY;
    do {
        ++top;
    CHILD_LOOP_ENTRY:
        uint32_t k = tzcnt(mask);
        __m128 entry, tTempDelta = _mm_load_ss(frame->tDeltaMem + k);
        auto i = top;
        goto PUSH_LOOP_ENTRY;
        do {
            store(&(frame->cullStack + i)->tDelta, entry);
            --i;
        PUSH_LOOP_ENTRY:
            entry = load_m128(&(frame->cullStack - 1 + i)->tDelta);
        } while (_mm_comigt_ss(entry, tTempDelta));
        (frame->cullStack + i)->node = node + node->boxData.children.offset[k];
        _mm_store_ss(&(frame->cullStack + i)->tDelta, tTempDelta);
        (frame->cullStack + i)->tMinTemp = frame->tMin[k];
        mask = clearLowestBit(mask);
    } while (mask);
    ++top;
    goto POP;
}

static __forceinline uint32_t tileCull3D(uint32_t* triIndices,
                                         uint32_t maxTriCount,
                                         StackFrameTile* frame,
                                         uint32_t top,
                                         const RayPacketFrustum3D& frustum) {
    auto negateMask = m256(-0.0f);
    auto tMin = 0.0f;
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
    auto node = (frame->stack + top)->node;
    tMin = (frame->stack + top)->tMin;

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
    if ((frame->stack + top)->mask)
        goto POP;

    // conservative min ray bundle distance to children along the primary traversal axis (stick it into a temp)
    store(frame->tMin,
          _mm_xor_ps(m128(negateMask), load_m128((float*)((uintptr_t)node + frustum.distanceEstimateOffset))));

    auto k = tzcnt(mask);        // index of first hit child node
                                 // If I only hit one child, avoid stack
    if (!clearLowestBit(mask)) { // only one bit set?
        // if we only hit one child node, avoid inserting into the stack. Directly replace the current node
        // with the hit node, then jump to testing it.
        tMin = frame->tMin[k];
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
        __m128 tMinTemp = _mm_load_ss(frame->tMin + k); // grab the temp distance estimate we stored earlier
        // search for an insertion point, bubbling up existing entries with smaller distances
        // such that the top of the stack has the element with the least distance
        __m128 entry;
        auto i = top;
        goto PUSH_LOOP_ENTRY;
        do {
            store(&(frame->stack + i)->tMin, entry); // this is a 128-bit store containing a whole stack entry
            --i;
        PUSH_LOOP_ENTRY:
            entry = load_m128(&(frame->stack - 1 + i)->tMin);
        } while (_mm_comilt_ss(entry, tMinTemp));

        // we've found our insertion point, place the child there
        (frame->stack + i)->node = node + node->boxData.children.offset[k];
        (frame->stack + i)->mask = 0;
        _mm_store_ss(&(frame->stack + i)->tMin, tMinTemp);

        mask = clearLowestBit(mask);
    } while (mask);

    goto POP_LAST;
}

// This could very easily be drastically simplified. But it is debugging code...
// http://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
static double angleOnSphere(vector3 A, vector3 B, vector3 C) {
    vector3 AcrossB = cross(A, B);
    vector3 CcrossB = cross(C, B);
    return acosf(dot(AcrossB, CcrossB) / (length(AcrossB) * length(CcrossB)));
}

// http://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
static double sphericalTriangleArea(vector3 A, vector3 B, vector3 C) {
    return angleOnSphere(A, B, C) + angleOnSphere(B, C, A) + angleOnSphere(C, A, B) - M_PI;
}

// Only works for frusta with center of projection
static double solidAngle(RayPacketFrustum3D frustum) {
    vector3 v[4] = {normalize(frustum.pointDir[0]), normalize(frustum.pointDir[1]), normalize(frustum.pointDir[2]),
                    normalize(frustum.pointDir[3])};

    return sphericalTriangleArea(v[0], v[1], v[2]) + sphericalTriangleArea(v[0], v[2], v[3]);
}

static vector4 minMaxMeanMedian(std::vector<double>& vec, size_t& validCount) {
    std::vector<double> cleanVec;
    for (int i = 0; i < vec.size(); ++i) {
        if (!isnan(vec[i])) {
            cleanVec.push_back(vec[i]);
        }
    }
    validCount = cleanVec.size();
    // printf("valid/total = %zd/%zd = %f\n", cleanVec.size(), vec.size(), cleanVec.size() / (double)vec.size());
    double minV = INFINITY;
    double maxV = -INFINITY;
    double sumV = 0.0f;
    std::sort(cleanVec.begin(), cleanVec.end());
    for (auto d : cleanVec) {
        sumV += d;
        minV = min(minV, d);
        maxV = max(maxV, d);
    }
    return vector4(float(minV), float(maxV), float(sumV) / cleanVec.size(), float(cleanVec[cleanVec.size() / 2]));
};

struct TaskData {
    uint32_t triIndexCount;
    std::vector<unsigned> tileIndexRemapOccupied;
    std::vector<unsigned> tileIndexRemapEmpty;
    std::vector<TileTriRange> tileTriRanges;
    ArrayView<unsigned> triIndices;
    void reset(ArrayView<unsigned> fullTriIndices, size_t bufferStart, size_t bufferEnd) {
        triIndexCount = 0;
        triIndices = ArrayView<unsigned>(fullTriIndices.data() + bufferStart, bufferEnd - bufferStart);
        tileIndexRemapOccupied.clear();
        tileIndexRemapEmpty.clear();
        tileTriRanges.clear();
    }
};

static void generatePerChunkTriangleListsOneThread(
    const BlockInfo& blockInfo, uint32_t startBlock, uint32_t endBlock, const BVHNode* nodes, TaskData* perThread) {
#if DEBUG_STATS
    auto startTime = (double)__rdtsc();
#endif

    if (startBlock == endBlock) {
        return;
    }

    perThread->triIndexCount = 0;

    StackFrameBlock frameBlock;
    for (uint32_t b = startBlock; b < endBlock; ++b) {
        const RayPacketFrustum3D& blockFrustum = blockInfo.blockFrusta[b];
        uint32_t stackSize = blockCull3D(&frameBlock, nodes, blockFrustum);

        if (!stackSize) { // we hit nothing?
            for (unsigned i = 0; i < TILES_PER_BLOCK; ++i) {
                auto globalTileIndex = b * TILES_PER_BLOCK + i;
                perThread->tileIndexRemapEmpty.push_back(globalTileIndex);
            }
            continue;
        }

        frameBlock.sort(stackSize);

        auto i = TILES_PER_BLOCK;
        do {
            auto tileIndex = (TILES_PER_BLOCK - i);
            auto globalTileIndex = b * TILES_PER_BLOCK + tileIndex;

            StackFrameTile frameTile;
            for (uint32_t slot = 0; slot != stackSize; ++slot)
                store(&(frameTile.stack + slot)->tMin, load_m128(&(frameBlock.sortedStack + slot)->tMin));

            uint32_t* triIndices = perThread->triIndices.data() + perThread->triIndexCount;
            uint32_t maxTriCount = uint32_t(perThread->triIndices.size()) - perThread->triIndexCount;

            const RayPacketFrustum3D& tileFrustum = blockInfo.tileFrusta[globalTileIndex];
            uint32_t outputTriCount = tileCull3D(triIndices, maxTriCount, &frameTile, stackSize, tileFrustum);

            if (outputTriCount) {
                TileTriRange triRange;
                triRange.start = perThread->triIndexCount;
                triRange.end = triRange.start + outputTriCount;

                perThread->tileTriRanges.push_back(triRange);
                perThread->tileIndexRemapOccupied.push_back(globalTileIndex);
                perThread->triIndexCount += outputTriCount;
            } else {
                perThread->tileIndexRemapEmpty.push_back(globalTileIndex);
            }
        } while (--i);
    }

#if DEBUG_STATS
    // double deltaTimeMs = ((double)__rdtsc() - startTime) * whunt::gRcpCPUFrequency * 1000.0;
    std::vector<double> blockFrustaAngle(endBlock - startBlock);
    std::vector<double> tileFrustaAngle((endBlock - startBlock) * TILES_PER_BLOCK);
    for (size_t b = startBlock; b < endBlock; ++b) {
        blockFrustaAngle[b - startBlock] = solidAngle(blockInfo.blockFrusta[b]);
        for (size_t t = 0; t < TILES_PER_BLOCK; ++t) {
            tileFrustaAngle[t] = solidAngle(blockInfo.tileFrusta[b * TILES_PER_BLOCK + t]);
        }
    }
    size_t validBlocks, validTiles;
    vector4 m4Block = minMaxMeanMedian(blockFrustaAngle, validBlocks);
    vector4 m4Tile = minMaxMeanMedian(tileFrustaAngle, validTiles);

    // printf("---- Block cull [%u,%u) solid angle:  time %f, %u triangles,  %g percent coverage\n", startBlock,
    // endBlock, deltaTimeMs, currentTriIdx, 100.0*m4X.z*validBlocks / (4 * M_PI));
    printf("%u, %u, %g, %g\n", startBlock, perThread->triIndexCount, 100.0 * m4Block.z * validBlocks / (4 * M_PI),
           100.0 * m4Tile.z * validTiles / (4 * M_PI));
#endif
}

void Raycaster::generatePerChunkTriangleListsParallel(const BlockInfo& blockInfo, Camera_StreamedData* streamed) {
    const BVHNode* nodes = _nodes.data();
    ArrayView<uint32_t> triIndices(streamed->triIndices.dataHost(), streamed->triIndices.size());

#if DEBUG_STATS
    std::vector<double> blockFrustaAngle(blockInfo.blockFrusta.size());
    for (int i = 0; i < blockInfo.blockFrusta.size(); ++i) {
        blockFrustaAngle[i] = solidAngle(blockInfo.blockFrusta[i]);
    }
    size_t validBlocks;
    vector4 m4X = minMaxMeanMedian(blockFrustaAngle, validBlocks);
    printf("Block: Min,Max,Mean,Median: %g, %g, %g, %g\n", m4X.x, m4X.y, m4X.z, m4X.w);
    printf("Percent of sphere covered by block frusta: %g\n", 100.0 * m4X.z * validBlocks / (4 * M_PI));

    std::vector<double> tileFrustaAngle(blockInfo.tileFrusta.size());
    for (int i = 0; i < blockInfo.tileFrusta.size(); ++i) {
        // Convert to square degrees from steradians
        tileFrustaAngle[i] = solidAngle(blockInfo.tileFrusta[i]);
    }
    size_t validTiles;
    m4X = minMaxMeanMedian(tileFrustaAngle, validTiles);
    printf("Tile: Min,Max,Mean,Median: %g, %g, %g, %g\n", m4X.x, m4X.y, m4X.z, m4X.w);
    printf("Percent of sphere covered by tile frusta: %g\n", 100.0 * m4X.z * validTiles / (4 * M_PI));

#endif

#if DEBUG_STATS || TIME_BLOCK_CULL
    Timer timer;
#endif

    enum { maxTasks = 4096 };
    // workload per task
    // 1 is too low, and incurs overhead from switching tasks too frequently inside the thread pool
    // 2 seems the fastest (though this could vary depending on the scene and sample distribution)
    // 3+ seems to become less efficient due to workload balancing
    enum { blocksPerThread = 2 };
    uint32_t blockCount = uint32_t(blockInfo.blockFrusta.size());
    uint32_t numTasks = (blockCount + blocksPerThread - 1) / blocksPerThread;
    assert(numTasks <= maxTasks);
    numTasks = min<uint32_t>(maxTasks, numTasks);
    uint32_t triSpacePerThread = uint32_t(triIndices.size() / numTasks);
    assert(triSpacePerThread * numTasks <= triIndices.size());

    std::future<void> taskResults[maxTasks];
    static TaskData taskData[maxTasks];
    for (uint32_t i = 0; i < numTasks; ++i) {
        uint32_t startTriIndex = i * triSpacePerThread;
        uint32_t endTriIndex = startTriIndex + triSpacePerThread;
        taskData[i].reset(triIndices, startTriIndex, endTriIndex);

        uint32_t startBlock = min(blockCount, i * blocksPerThread);
        uint32_t endBlock = min(blockCount, (i + 1) * blocksPerThread);
        if (i == numTasks - 1)
            assert(endBlock == blockCount);

        taskResults[i] = _threadPool->addTask(generatePerChunkTriangleListsOneThread, blockInfo, startBlock, endBlock,
                                              nodes, &taskData[i]);
    }

#if DEBUG_STATS
    uint64_t triIndexCount = 0;
    uint32_t maxTaskTriCount = 0;
#endif
    uint32_t tileTriOffsetsStreamed = 0;
    uint32_t* streamTileIndexRemapEmpty = streamed->tileIndexRemapEmpty.dataHost();
    uint32_t* streamTileIndexRemapOccupied = streamed->tileIndexRemapOccupied.dataHost();
    TileTriRange* streamTileTriRanges = streamed->tileTriRanges.dataHost();
    for (uint32_t taskIndex = 0; taskIndex < numTasks; taskIndex++) {
        taskResults[taskIndex].get();
        const TaskData& task = taskData[taskIndex];

        for (auto emptyTileIndex : task.tileIndexRemapEmpty) {
            streamTileIndexRemapEmpty[streamed->tileCountEmpty++] = emptyTileIndex;
        }

        for (auto occupiedTileIndex : task.tileIndexRemapOccupied) {
            streamTileIndexRemapOccupied[streamed->tileCountOccupied++] = occupiedTileIndex;
        }

        // tileTriRange within the task is relative to the task's smaller view of the buffer and needs to be offset
        uint32_t taskTriOffset = uint32_t(task.triIndices.data() - streamed->triIndices.dataHost());
        for (auto tileTriRange : task.tileTriRanges) {
            TileTriRange triRangeGlobal;
            triRangeGlobal.start = taskTriOffset + tileTriRange.start;
            triRangeGlobal.end = taskTriOffset + tileTriRange.end;
            streamTileTriRanges[tileTriOffsetsStreamed++] = triRangeGlobal;
        }

#if DEBUG_STATS
        triIndexCount += task.triIndexCount;
        maxTaskTriCount = max(maxTaskTriCount, task.triIndexCount);
#endif
    }

#if DEBUG_STATS || TIME_BLOCK_CULL
    double deltaTime = timer.get();
    static double minDeltaTime = DBL_MAX;
    minDeltaTime = min(minDeltaTime, deltaTime);

    static uint64_t frameIndex = 0;
    enum { profileFrameSkip = 64 };
    if (frameIndex % profileFrameSkip == 0) {
        printf("---- Block cull time: %.2fms, min %.2fms\n", deltaTime * 1000.0, minDeltaTime * 1000.0);
    }
    frameIndex++;
#endif
#if DEBUG_STATS
    printf("Total Triangle Idx Count %u\n", triIndexCount);
    printf("Max Triangle Idx Count Per Task %u\n", maxTaskTriCount);
#endif
}

} // namespace hvvr
