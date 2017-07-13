#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "bvh_node.h"
#include "graphics_types.h"
#include "magic_constants.h"
#include "raycaster_common.h"
#include "samples.h"
#include "vector_math.h"
#include "dynamic_array.h"

#include <assert.h>


namespace hvvr {

class Raycaster;

// Encoding of a frustum for ray packet traversal
struct RayPacketFrustum2D {
    // xMin, xNegMax, yMin, yNegMax
    vector4 data;

    RayPacketFrustum2D(float xMin, float xMax, float yMin, float yMax) : data(xMin, -xMax, yMin, -yMax) {}
    RayPacketFrustum2D() = default;

    // Set the mins to infinity and maxs to -infinity
    void setEmpty() {
        data = vector4(std::numeric_limits<float>::infinity());
    }
    void merge(float x, float y) {
        data = min(data, vector4(x, -x, y, -y));
    }
    void merge(const RayPacketFrustum2D& other) {
        data = min(data, other.data);
    }
    void intersect(const RayPacketFrustum2D& other) {
        data = max(data, other.data);
    }

    inline float xMin() const {
        return data.x;
    }
    inline float xMax() const {
        return -data.y;
    }
    inline float yMin() const {
        return data.z;
    }
    inline float yMax() const {
        return -data.w;
    }
    inline float xNegMax() const {
        return data.y;
    }
    inline float yNegMax() const {
        return data.w;
    }
};

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
struct RayPacketFrustum3D {
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
    vector3 pointOrigin[pointCount];
    vector3 pointDir[pointCount];

    RayPacketFrustum3D() = default;
    // construct from four rays
    // o = ray origin, d = ray direction
    RayPacketFrustum3D(const vector3& o0,
                       const vector3& d0,
                       const vector3& o1,
                       const vector3& d1,
                       const vector3& o2,
                       const vector3& d2,
                       const vector3& o3,
                       const vector3& d3) {
        // remember these for later intersection tests
        pointOrigin[0] = o0;
        pointOrigin[1] = o1;
        pointOrigin[2] = o2;
        pointOrigin[3] = o3;

        pointDir[0] = d0;
        pointDir[1] = d1;
        pointDir[2] = d2;
        pointDir[3] = d3;

        updatePrecomputed();
    }

    RayPacketFrustum3D(const SimpleRayFrustum& f) {
        for (int i = 0; i < 4; ++i) {
            pointOrigin[i] = vector3(f.origins[i].x, f.origins[i].y, f.origins[i].z);
            pointDir[i] = vector3(f.directions[i].x, f.directions[i].y, f.directions[i].z);
        }
        updatePrecomputed();
    }

    void updatePrecomputed();

    void setEmpty() {
        assert(false);
    }

    void merge(float x, float y) {
        (void)x;
        (void)y;
        assert(false);
    }
    void merge(const RayPacketFrustum3D& other) {
        (void)other;
        assert(false);
    }
    void intersect(const RayPacketFrustum3D& other) {
        (void)other;
        assert(false);
    }

    // TODO(anankervis):
    // -refactor plane representation such that dot(p, vector4(v, 1)) == 0 (negate the W component)
    // -transform() and updatePrecomputed() use mInvTranspose * plane to update plane equations (inc axes of sep)
    // --as opposed to recomputing all the cross products and such, use transpose of inverse for normals/planes
    RayPacketFrustum3D transform(const matrix4x4& m, const matrix4x4& mInvTranspose) const {
        (void)mInvTranspose;
        return RayPacketFrustum3D(
            vector3(m * vector4(pointOrigin[0], 1.0f)), matrix3x3(m) * pointDir[0],
            vector3(m * vector4(pointOrigin[1], 1.0f)), matrix3x3(m) * pointDir[1],
            vector3(m * vector4(pointOrigin[2], 1.0f)), matrix3x3(m) * pointDir[2],
            vector3(m * vector4(pointOrigin[3], 1.0f)), matrix3x3(m) * pointDir[3]);
    }

    // intersect against the four child AABBs, corresponding bit index is set if the AABB passes
    // This test is designed to be fast and conservative. Follow with testBVHNodeChildrenRefine
    // to reject additional cases.
    __forceinline unsigned int testBVHNodeChildren(const BVHNode& node) const;
    // intersect against the four child AABBs, corresponding bit index is set if the AABB passes
    // This refines the initial results from testBVHNodeChildren by running additional tests.
    __forceinline unsigned int testBVHNodeChildrenRefine(const BVHNode& node, unsigned int result) const;

    bool testPoint(const vector3& p) const {
        for (int planeIndex = 0; planeIndex < planeCount; planeIndex++) {
            if (dot(vector3(plane[planeIndex]), p) - plane[planeIndex].w > 0)
                return false;
        }

        return true;
    }
};

struct BlockInfo {
    ArrayView<const RayPacketFrustum3D> blockFrusta;
    ArrayView<const RayPacketFrustum3D> tileFrusta;
};

} // namespace hvvr
