#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "traverse_avx.h"
#include "avx.h"
#include "bvh_node.h"

#include <float.h>
#include <stddef.h>

namespace hvvr { namespace traverse { namespace avx {

// TODO(anankervis): optimize this - it gets called a lot
// find the major axis, and precompute some values for the intersection tests
Frustum::Frustum(const vector3* origin, const vector3* dir) {
    static_assert(planeCount == pointCount, "Frustum init code assumes planeCount == pointCount");
    vector3 avgDir(0.0f);
    for (int p = 0; p < planeCount; p++) {
        pointOrigin[p] = origin[p];
        pointDir[p] = dir[p];

        // TODO: This doesn't really belong here, it's a hack to enable discarding frusta culled by
        // clipping planes earlier in the pipeline... ideally, those frusta shouldn't make it this far.
        // If we have an infinite orgin, assume a fully degenerate frustum that encompasses nothing.
        if (isinf(pointOrigin[0].x)) {
            plane[p] = vector4(0, 0, 0, -INFINITY);
        } else {
            plane[p] = vector4(cross(dir[p], dir[(p + 1) % planeCount]), 0);
            plane[p].w = dot(origin[p], vector3(plane[p]));
        }

        avgDir += dir[p];
    }

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
        __m256 swizzledPointOriginX = -m256(origin[0].x, origin[1].x, origin[2].x, origin[3].x,
                                            -origin[0].x, -origin[1].x, -origin[2].x, -origin[3].x);
        __m256 swizzledPointOriginY = -m256(origin[0].y, origin[1].y, origin[2].y, origin[3].y,
                                            -origin[0].y, -origin[1].y, -origin[2].y, -origin[3].y);
        __m256 swizzledPointOriginZ = -m256(origin[0].z, origin[1].z, origin[2].z, origin[3].z,
                                            -origin[0].z, -origin[1].z, -origin[2].z, -origin[3].z);

        // prep for computing max in the lower 4 floats, -min in the upper 4
        // baking in a hi/lo 128-bit shuffle (a negate)
        // also baking in a max with zero, and scaling by FLT_MAX
        // This will be used to extend the frustum's projection to infinity in the direction
        // the frustum is pointing (the frustum has no far plane).
        __m256 swizzledPointDirX =
            m256(FLT_MAX) *
            _mm256_max_ps(_mm256_setzero_ps(), -m256(dir[0].x, dir[1].x, dir[2].x, dir[3].x,
                                                     -dir[0].x, -dir[1].x, -dir[2].x, -dir[3].x));
        __m256 swizzledPointDirY =
            m256(FLT_MAX) *
            _mm256_max_ps(_mm256_setzero_ps(), -m256(dir[0].y, dir[1].y, dir[2].y, dir[3].y,
                                                     -dir[0].y, -dir[1].y, -dir[2].y, -dir[3].y));
        __m256 swizzledPointDirZ =
            m256(FLT_MAX) *
            _mm256_max_ps(_mm256_setzero_ps(), -m256(dir[0].z, dir[1].z, dir[2].z, dir[3].z,
                                                     -dir[0].z, -dir[1].z, -dir[2].z, -dir[3].z));

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
            cross(dir[0], vector3(1, 0, 0)), cross(dir[0], vector3(0, 1, 0)),
            cross(dir[0], vector3(0, 0, 1)),

            cross(dir[1], vector3(1, 0, 0)), cross(dir[1], vector3(0, 1, 0)),
            cross(dir[1], vector3(0, 0, 1)),

            cross(dir[2], vector3(1, 0, 0)), cross(dir[2], vector3(0, 1, 0)),
            cross(dir[2], vector3(0, 0, 1)),

            cross(dir[3], vector3(1, 0, 0)), cross(dir[3], vector3(0, 1, 0)),
            cross(dir[3], vector3(0, 0, 1)),
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
                    _mm256_fmadd_ps(edgeX, _mm256_broadcast_ss((float*)&origin[pointIndex].x),
                                    _mm256_fmadd_ps(edgeY, _mm256_broadcast_ss((float*)&origin[pointIndex].y),
                                                    edgeZ * _mm256_broadcast_ss((float*)&origin[pointIndex].z)));
                minFrustum = min(minFrustum, d0);
                maxFrustum = max(maxFrustum, d0);

                // frustum has no far plane, it extends to infinity
                __m256 d1 =
                    _mm256_fmadd_ps(edgeX, _mm256_broadcast_ss((float*)&dir[pointIndex].x),
                                    _mm256_fmadd_ps(edgeY, _mm256_broadcast_ss((float*)&dir[pointIndex].y),
                                                    edgeZ * _mm256_broadcast_ss((float*)&dir[pointIndex].z)));

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

}}} // namespace hvvr::traverse::avx
