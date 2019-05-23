#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "traverse_ref.h"

#include <float.h>
#include <stddef.h>

namespace hvvr { namespace traverse { namespace ref {

Frustum::Frustum(const vector3* origin, const vector3* dir) {
    // If we have an infinite orgin, assume a fully degenerate frustum that encompasses nothing.
    degenerate = isinf(origin[0].x);

    static_assert(planeCount == pointCount, "Frustum init code assumes planeCount == pointCount");
    vector3 avgDir(0.0f);
    for (int p = 0; p < planeCount; p++) {
        pointOrigin[p] = origin[p];
        pointDir[p] = dir[p];

        plane[p] = vector4(cross(dir[p], dir[(p + 1) % planeCount]), 0);
        plane[p].w = dot(origin[p], vector3(plane[p]));

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

    // project the frustum onto the X, Y, Z axes
    // The frustum's four origin points will form one end of the projection, and the projection
    // will extend to infinity on the other end, depending on the sign of the four
    // edge directions. It is possible for a wide frustum to extend to infinity on
    // both ends of a projection.
    projMin = vector3(FLT_MAX, FLT_MAX, FLT_MAX);
    projMax = vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int v = 0; v < pointCount; v++) {
        if (dir[v].x < 0) {
            projMin.x = -FLT_MAX;
            projMax.x = max(projMax.x, origin[v].x);
        } else if (dir[v].x > 0) {
            projMin.x = min(projMin.x, origin[v].x);
            projMax.x = FLT_MAX;
        } else {
            projMin.x = min(projMin.x, origin[v].x);
            projMax.x = max(projMax.x, origin[v].x);
        }

        if (dir[v].y < 0) {
            projMin.y = -FLT_MAX;
            projMax.y = max(projMax.y, origin[v].y);
        } else if (dir[v].y > 0) {
            projMin.y = min(projMin.y, origin[v].y);
            projMax.y = FLT_MAX;
        } else {
            projMin.y = min(projMin.y, origin[v].y);
            projMax.y = max(projMax.y, origin[v].y);
        }

        if (dir[v].z < 0) {
            projMin.z = -FLT_MAX;
            projMax.z = max(projMax.z, origin[v].z);
        } else if (dir[v].z > 0) {
            projMin.z = min(projMin.z, origin[v].z);
            projMax.z = FLT_MAX;
        } else {
            projMin.z = min(projMin.z, origin[v].z);
            projMax.z = max(projMax.z, origin[v].z);
        }
    }

    // precomputation for the AABB x frustum edge tests
    {
        refineEdge[ 0] = cross(dir[0], vector3(1, 0, 0));
        refineEdge[ 1] = cross(dir[0], vector3(0, 1, 0));
        refineEdge[ 2] = cross(dir[0], vector3(0, 0, 1));

        refineEdge[ 3] = cross(dir[1], vector3(1, 0, 0));
        refineEdge[ 4] = cross(dir[1], vector3(0, 1, 0));
        refineEdge[ 5] = cross(dir[1], vector3(0, 0, 1));

        refineEdge[ 6] = cross(dir[2], vector3(1, 0, 0));
        refineEdge[ 7] = cross(dir[2], vector3(0, 1, 0));
        refineEdge[ 8] = cross(dir[2], vector3(0, 0, 1));

        refineEdge[ 9] = cross(dir[3], vector3(1, 0, 0));
        refineEdge[10] = cross(dir[3], vector3(0, 1, 0));
        refineEdge[11] = cross(dir[3], vector3(0, 0, 1));

        // project the frustum onto the edge axes
        // (this is essentially the same math as the above projection onto the AABB axes)
        for (int e = 0; e < edgeCount; e++) {
            refineMin[e] = FLT_MAX;
            refineMax[e] = -FLT_MAX;
            for (int v = 0; v < pointCount; v++) {
                float d0 = dot(refineEdge[e], origin[v]);
                refineMin[e] = min(refineMin[e], d0);
                refineMax[e] = max(refineMax[e], d0);

                // let's not project the frustum to infinity over something that would normally be a rounding error
                float epsilon = .000001f;

                float d1 = dot(refineEdge[e], dir[v]);
                if (d1 < -epsilon)
                    refineMin[e] = -FLT_MAX;
                else if (d1 > epsilon)
                    refineMax[e] = FLT_MAX;
            }
        }
    }
}

uint32_t Frustum::testBVHNodeChildren(const BVHNode& node) const {
    uint32_t result = childMaskAll;

    vector3 aabbMin[childCount];
    vector3 aabbMax[childCount];
    for (int c = 0; c < childCount; c++) {
        aabbMin[c].x = -node.xNegMin[c];
        aabbMin[c].y = -node.yNegMin[c];
        aabbMin[c].z = -node.zNegMin[c];

        aabbMax[c].x = node.xMax[c];
        aabbMax[c].y = node.yMax[c];
        aabbMax[c].z = node.zMax[c];
    }

    // test AABB against frustum faces
    // n-vertex of AABB for this plane - the most negative along the plane normal
    // n-vertex (and p-vertex) is a permutation of AABB min and max components
    for (int p = 0; p < planeCount; p++) {
        for (int c = 0; c < childCount; c++) {
            vector3 nPoint(
                plane[p].x < 0 ? aabbMax[c].x : aabbMin[c].x,
                plane[p].y < 0 ? aabbMax[c].y : aabbMin[c].y,
                plane[p].z < 0 ? aabbMax[c].z : aabbMin[c].z);

            float d = dot(vector3(plane[p]), nPoint);

            if (d > plane[p].w) {
                result &= ~(1 << c); // child culled by frustum face
            }
        }
    }

    // test frustum against AABB faces
    // check for overlap of frustum's projection onto AABB axes
    for (int c = 0; c < childCount; c++) {
        if (projMax.x < aabbMin[c].x || projMin.x > aabbMax[c].x ||
            projMax.y < aabbMin[c].y || projMin.y > aabbMax[c].y ||
            projMax.z < aabbMin[c].z || projMin.z > aabbMax[c].z) {
            result &= ~(1 << c); // child culled due to no overlap
        }
    }

    return result;
}

uint32_t Frustum::testBVHNodeChildrenRefine(const BVHNode& node, uint32_t prevResult) const {
    uint32_t result = prevResult;

    vector3 aabbMin[childCount];
    vector3 aabbMax[childCount];
    for (int c = 0; c < childCount; c++) {
        aabbMin[c].x = -node.xNegMin[c];
        aabbMin[c].y = -node.yNegMin[c];
        aabbMin[c].z = -node.zNegMin[c];

        aabbMax[c].x = node.xMax[c];
        aabbMax[c].y = node.yMax[c];
        aabbMax[c].z = node.zMax[c];
    }

    // test the remaining axes of separation, the cross products of
    // unique edge direction combinations from the frustum and AABB
    for (int e = 0; e < edgeCount; e++) {
        for (int c = 0; c < childCount; c++) {
            // find the p-vertex
            vector3 pPoint(
                refineEdge[e].x >= 0 ? aabbMax[c].x : aabbMin[c].x,
                refineEdge[e].y >= 0 ? aabbMax[c].y : aabbMin[c].y,
                refineEdge[e].z >= 0 ? aabbMax[c].z : aabbMin[c].z);

            // find the n-vertex
            vector3 nPoint(
                refineEdge[e].x <  0 ? aabbMax[c].x : aabbMin[c].x,
                refineEdge[e].y <  0 ? aabbMax[c].y : aabbMin[c].y,
                refineEdge[e].z <  0 ? aabbMax[c].z : aabbMin[c].z);

            float dMax = dot(refineEdge[e], pPoint);
            float dMin = dot(refineEdge[e], nPoint);

            if (dMax < refineMin[e] || dMin > refineMax[e]) {
                result &= ~(1 << c); // child culled due to no overlap along the edge axis
            }
        }
    }

    return result;
}

}}} // namespace hvvr::traverse::ref
