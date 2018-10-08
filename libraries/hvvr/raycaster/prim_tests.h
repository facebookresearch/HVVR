#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gpu_samples.h"
#include "graphics_types.h"
#include "vector_math.h"


namespace hvvr {

struct FrustumPlanes {
    enum { planeCount = 4 };
    vector4 p[planeCount];

    CUDA_DEVICE FrustumPlanes(const SimpleRayFrustum& rays) {
        for (int n = 0; n < planeCount; n++) {
            vector3 dir = cross(rays.directions[(n + 0) % planeCount], rays.directions[(n + 1) % planeCount]);
            float dist = dot(rays.origins[n], dir);
            p[n] = vector4(dir, dist);
        }
    }

    CUDA_DEVICE bool test(const PrecomputedTriangleIntersect& tri) {
        vector3 v0 = tri.v0;
        vector3 edge0 = tri.edge0;
        vector3 edge1 = tri.edge1;

        for (int n = 0; n < planeCount; n++) {
            vector3 dir(p[n]);
            float dist = p[n].w;

            float d0 = dot(dir, v0);
            float d1 = d0 + dot(dir, edge0);
            float d2 = d0 + dot(dir, edge1);

            if (d0 > dist && d1 > dist && d2 > dist) {
                return false;
            }
        }

        return true;
    }
};

// return value xyz are barycentrics, w is t
template <bool CAN_FAIL>
CUDA_DEVICE_INL vector4 TestRayTriangle(vector3 rayOrigin, vector3 rayDir, const PrecomputedTriangleIntersect& tri) {
    vector3 v0 = tri.v0;
    vector3 edge0 = tri.edge0;
    vector3 edge1 = tri.edge1;

    vector3 n = cross(edge0, edge1);

    float denom = dot(-rayDir, n);
    // ray is parallel to triangle, or fails backfacing test?
    if (CAN_FAIL && denom <= 0.0f)
        return vector4(-1.0f, -1.0f, -1.0f, -1.0f);
    vector3 v0ToRayOrigin = rayOrigin - v0;
    float t = dot(v0ToRayOrigin, n);
    if (CAN_FAIL && t < 0.0f) // intersection falls before ray origin?
        return vector4(-1.0f, -1.0f, -1.0f, -1.0f);

    // compute barycentrics
    vector3 e = cross(-rayDir, v0ToRayOrigin);
    float v = dot(edge1, e);
    float w = -dot(edge0, e);

    float ood = 1.0f / denom;
    t *= ood;
    v *= ood;
    w *= ood;
    float u = 1.0f - v - w;

    return vector4(u, v, w, t);
}

enum IntersectResult { intersect_all_out = 0, intersect_all_in, intersect_partial };

// assumes tris where denom <= 0.0f and t < 0.0f have already been rejected
// TODO(anankervis): optimize this... if the whole tile shares a coord system, could compute
// min/max U, V, W directly from tile-uniform derivatives
CUDA_DEVICE_INL IntersectResult TestTriangleFrustaUVW(const SimpleRayFrustum& rays,
                                                      const PrecomputedTriangleIntersect& tri) {
    vector4 uvw[4];
    for (int n = 0; n < 4; n++) {
        uvw[n] = TestRayTriangle<false>(rays.origins[n], rays.directions[n], tri);
    }

    float uMin = FLT_MAX;
    float uMax = -FLT_MAX;
    float vMin = FLT_MAX;
    float vMax = -FLT_MAX;
    float wMin = FLT_MAX;
    float wMax = -FLT_MAX;
    for (int n = 0; n < 4; n++) {
        uMin = min(uMin, uvw[n].x);
        uMax = max(uMax, uvw[n].x);
        vMin = min(vMin, uvw[n].y);
        vMax = max(vMax, uvw[n].y);
        wMin = min(wMin, uvw[n].z);
        wMax = max(wMax, uvw[n].z);
    }

    if (uMax < 0 || uMin > 1 || vMax < 0 || vMin > 1 || wMax < 0 || wMin > 1)
        return intersect_all_out;
    if (uMin < 0 || uMax > 1 || vMin < 0 || vMax > 1 || wMin < 0 || wMax > 1)
        return intersect_partial;
    return intersect_all_in;
}

CUDA_DEVICE_INL void GetDifferentials(vector3 edge0,
                                      vector3 edge1,
                                      vector3 v0ToRayOrigin,
                                      vector3 majorDirDiff,
                                      vector3 minorDirDiff,
                                      vector2& dDenomDAlpha,
                                      vector2& dVdAlpha,
                                      vector2& dWdAlpha) {
    vector3 normal = cross(edge0, edge1);
    dDenomDAlpha = vector2(dot(-majorDirDiff, normal), dot(-minorDirDiff, normal));

    dVdAlpha =
        vector2(dot(edge1, cross(-majorDirDiff, v0ToRayOrigin)), dot(edge1, cross(-minorDirDiff, v0ToRayOrigin)));

    dWdAlpha =
        vector2(-dot(edge0, cross(-majorDirDiff, v0ToRayOrigin)), -dot(edge0, cross(-minorDirDiff, v0ToRayOrigin)));
}

// tile-wide values
template <bool DoF>
struct IntersectTriangleTile;

template <>
struct IntersectTriangleTile<false> {
    float t;
    vector3 edge0;
    vector3 edge1;
    vector3 v0ToRayOrigin;

    // returns false if no rays originating at rayOrigin can intersect the triangle's plane (assumes backface testing)
    CUDA_DEVICE IntersectResult setup(const PrecomputedTriangleIntersect& triPrecomp, vector3 rayOrigin) {
        vector3 v0 = triPrecomp.v0;
        edge0 = triPrecomp.edge0;
        edge1 = triPrecomp.edge1;

        vector3 normal = cross(edge0, edge1);
        v0ToRayOrigin = rayOrigin - v0;

        t = dot(v0ToRayOrigin, normal);

        // this test could be precomputed each frame for a pinhole camera, and failing triangles discarded earlier
        if (t < 0.0f)
            return intersect_all_out; // ray origin is behind the triangle's plane

        return intersect_all_in;
    }
};

// per-thread values
struct IntersectTriangleThread {
    float denomCenter;
    float vCenter;
    float wCenter;
    vector2 dDenomDAlpha;
    vector2 dVdAlpha;
    vector2 dWdAlpha;

    CUDA_DEVICE IntersectTriangleThread(const IntersectTriangleTile<false>& triTile,
                                        vector3 rayDirCenter,
                                        vector3 majorDirDiff,
                                        vector3 minorDirDiff) {
        vector3 normal = cross(triTile.edge0, triTile.edge1);
        denomCenter = dot(-rayDirCenter, normal);

        // compute scaled barycentrics
        vector3 eCenter = cross(-rayDirCenter, triTile.v0ToRayOrigin);
        vCenter = dot(triTile.edge1, eCenter);
        wCenter = -dot(triTile.edge0, eCenter);
        GetDifferentials(triTile.edge0, triTile.edge1, triTile.v0ToRayOrigin, majorDirDiff, minorDirDiff, dDenomDAlpha,
                         dVdAlpha, dWdAlpha);
    }

    // returns true if the ray intersects the triangle and the intersection distance is less than depth
    // also updates the value of depth
    CUDA_DEVICE bool test(const IntersectTriangleTile<false>& triTile, vector2 alpha, float& depth) {
        // it seems that the CUDA compiler is missing an opportunity to merge multiply + add across function calls into
        // FMA, so no call to dot product function here...
        // 2 FMA
        float denom = denomCenter + dDenomDAlpha.x * alpha.x + dDenomDAlpha.y * alpha.y;

        // t still needs to be divided by denom to get the correct distance
        // this is a combination of two tests:
        // 1) denom <= 0.0f // ray is parallel to triangle, or fails backfacing test
        // 2) tri.t >= depth * denom // failed depth test
        // tri.t is known to be >= 0.0f
        // depth is known to be >= 0.0f
        // we can safely test both conditions with only test #2
        // triTile.t >= depth * denom
        // 0.0f >= depth * denom - triTile.t
        // depth * denom - triTile.t < 0.0f
        // 1 FMA
        float depthDelta = depth * denom - triTile.t;

        // compute scaled barycentrics
        // 2 FMA
        float v = vCenter + dVdAlpha.x * alpha.x + dVdAlpha.y * alpha.y;
        // 2 FMA
        float w = wCenter + dWdAlpha.x * alpha.x + dWdAlpha.y * alpha.y;
        // 2 ADD
        float u = denom - v - w;

        // depth test from above, plus: u < 0.0f || v < 0.0f || w < 0.0f
        // 1 LOP3
        // 1 LOP
        int test = __float_as_int(depthDelta) | __float_as_int(u) | __float_as_int(v) | __float_as_int(w);
        // 1 ISETP
        if (test < 0)
            return false;

        // 1 RCP
        // 1 MUL
        depth = triTile.t * (1.0f / denom);
        return true;
    }

    CUDA_DEVICE void calcUVW(const IntersectTriangleTile<false>& triTile, vector2 alpha, vector3& uvw) {
        float denom = denomCenter + dDenomDAlpha.x * alpha.x + dDenomDAlpha.y * alpha.y;

        // compute scaled barycentrics
        float v = vCenter + dVdAlpha.x * alpha.x + dVdAlpha.y * alpha.y;
        float w = wCenter + dWdAlpha.x * alpha.x + dWdAlpha.y * alpha.y;
        float u = denom - v - w;

        float denomInv = 1.0f / denom;
        uvw.x = u * denomInv;
        uvw.y = v * denomInv;
        uvw.z = w * denomInv;
    }
};

// tile-wide values
template <>
struct IntersectTriangleTile<true> {
    vector2 dDenomDAlpha;

    vector3 edge0;
    vector3 edge1;

    float tCenter;
    float tU;
    float tV;

    vector3 v0ToLensCenter;

    // conservative test for backfacing and intersection point behind ray origin
    CUDA_DEVICE IntersectResult setup(const PrecomputedTriangleIntersect& triPrecomp,
                                      vector3 lensCenter,
                                      vector3 lensU,
                                      vector3 lensV) {
        vector3 v0 = triPrecomp.v0;
        edge0 = triPrecomp.edge0;
        edge1 = triPrecomp.edge1;

        vector3 normal = cross(edge0, edge1);

        v0ToLensCenter = lensCenter - v0;

        tCenter = dot(v0ToLensCenter, normal);
        tU = dot(lensU, normal);
        tV = dot(lensV, normal);
        float tExtent = fabsf(tU) + fabsf(tV);
        float tMax = tCenter + tExtent;

        if (tMax < 0.0f)
            return intersect_all_out; // ray origin is behind the triangle's plane for all lens positions

        dDenomDAlpha = vector2(dot(-lensU, normal), dot(-lensV, normal));

        float tMin = tCenter - tExtent;
        if (tMin < 0.0f)
            return intersect_partial; // ray origin is behind the triangle's plane for some lens positions

        return intersect_all_in; // all rays within the tile frusta will intersect the triangle's plane
    }
};

// per-thread values
struct IntersectTriangleThreadDoF {
    float denomCenter;

    CUDA_DEVICE IntersectTriangleThreadDoF(const IntersectTriangleTile<true>& triTile, vector3 rayDirCenter) {
        vector3 normal = cross(triTile.edge0, triTile.edge1);
        denomCenter = dot(-rayDirCenter, normal);
    }

    // returns true if the ray intersects the triangle and the intersection distance is less than depth
    // also updates the value of depth
    CUDA_DEVICE bool test(const IntersectTriangleTile<true>& triTile,
                          vector3 lensCenterToFocalCenter,
                          vector3 lensU,
                          vector3 lensV,
                          vector2 lensUV,
                          vector2 dirUV,
                          float& depth) {
        vector3 edge0 = triTile.edge0;
        vector3 edge1 = triTile.edge1;

        // 2 FMA
        float t = triTile.tCenter + triTile.tU * lensUV.x + triTile.tV * lensUV.y;

        // 2 FMA
        float denom = denomCenter + triTile.dDenomDAlpha.x * dirUV.x + triTile.dDenomDAlpha.y * dirUV.y;

        // t still needs to be divided by denom to get the correct distance
        // this is a combination of two tests:
        // 1) denom <= 0.0f // ray is parallel to triangle, or fails backfacing test
        // 2) tri.t >= depth * denom // failed depth test
        // tri.t is known to be >= 0.0f
        // depth is known to be >= 0.0f
        // we can safely test both conditions with only test #2
        // triTile.t >= depth * denom
        // 0.0f >= depth * denom - triTile.t
        // depth * denom - triTile.t < 0.0f
        // 1 FMA
        float depthDelta = depth * denom - t;

        // compute scaled barycentrics
        // 6 FMA
        vector3 v0ToRayOrigin = triTile.v0ToLensCenter + lensUV.x * lensU + lensUV.y * lensV;
        // 6 FMA
        vector3 rayDir = lensCenterToFocalCenter + dirUV.x * lensU + dirUV.y * lensV;
        // 3 MUL
        // 3 FMA
        vector3 e = cross(-rayDir, v0ToRayOrigin);
        // 1 MUL
        // 2 FMA
        float v = dot(edge1, e);
        // 1 MUL
        // 2 FMA
        float w = -dot(edge0, e);
        // 2 ADD
        float u = denom - v - w;

        // t < 0.0f || depth test from above || u < 0.0f || v < 0.0f || w < 0.0f
        // 2 LOP3
        // 1 LOP
        int test =
            __float_as_int(t) | __float_as_int(depthDelta) | __float_as_int(u) | __float_as_int(v) | __float_as_int(w);
        // 1 ISETP
        if (test < 0)
            return false;

        // 1 RCP
        // 1 MUL
        depth = t * (1.0f / denom);
        return true;
    }

    CUDA_DEVICE void calcUVW(const IntersectTriangleTile<true>& triTile,
                             vector3 lensCenterToFocalCenter,
                             vector3 lensU,
                             vector3 lensV,
                             vector2 lensUV,
                             vector2 dirUV,
                             vector3& uvw) {
        vector3 edge0 = triTile.edge0;
        vector3 edge1 = triTile.edge1;

        float denom = denomCenter + triTile.dDenomDAlpha.x * dirUV.x + triTile.dDenomDAlpha.y * dirUV.y;

        // compute scaled barycentrics
        vector3 v0ToRayOrigin = triTile.v0ToLensCenter + lensUV.x * lensU + lensUV.y * lensV;
        vector3 rayDir = lensCenterToFocalCenter + dirUV.x * lensU + dirUV.y * lensV;
        vector3 e = cross(-rayDir, v0ToRayOrigin);

        float v = dot(edge1, e);
        float w = -dot(edge0, e);
        float u = denom - v - w;

        float denomInv = 1.0f / denom;
        uvw.x = u * denomInv;
        uvw.y = v * denomInv;
        uvw.z = w * denomInv;
    }
};

} // namespace hvvr
