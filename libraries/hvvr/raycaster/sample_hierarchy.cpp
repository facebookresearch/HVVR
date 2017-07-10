/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "sample_hierarchy.h"
#include "kernel_constants.h"

#include <sstream>

#pragma warning(disable : 4505) // unreferenced local function has been removed

namespace hvvr {

static std::string toString(vector4 v) {
    std::ostringstream stringStream;
    stringStream << "vector4(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return stringStream.str();
}

// http://cseweb.ucsd.edu/~ravir/whitted.pdf outlines a basic technique for generating bounding frusta over a packet of
// non-point origin rays Step 1: Pick a major axis for the rays Step 2: Choose a near and far plane for the rays,
// perpendicular to the major axis Step 3: Compute AABB of the ray intersection points on both of the planes Step 4:
// Connect the corresponding corners of the AABBs with rays, these 4 rays define the corner rays of a conservative
// bounding frustum

// When converting from 2D samples to 3D, we can simplify this a bit
// Choose the major axis to be the Z axis. The near plane is z=0, the far plane can just be a camera parameter (negative
// Z) The AABB for the near plane is just the AABB of the lens.
static RayPacketFrustum3D get3DFrustumFrom2D(const RayPacketFrustum2D& frustum2D,
                                             const matrix3x3& sampleToCamera,
                                             ThinLens lens,
                                             float farPlane) {
    vector3 nearPoints[4];
    nearPoints[0] = vector3(-lens.radius, -lens.radius, 0);
    nearPoints[1] = vector3(+lens.radius, -lens.radius, 0);
    nearPoints[2] = vector3(+lens.radius, +lens.radius, 0);
    nearPoints[3] = vector3(-lens.radius, +lens.radius, 0);
    for (int i = 0; i < 4; ++i) {
        // printf("nearPoints[%d] = %s\n", i, toString(nearPoints[i]).c_str());
    }

    vector3 rayDirections[4];
    rayDirections[0] = sampleToCamera * vector3(frustum2D.xMin(), frustum2D.yMax(), 1);
    rayDirections[1] = sampleToCamera * vector3(frustum2D.xMax(), frustum2D.yMax(), 1);
    rayDirections[2] = sampleToCamera * vector3(frustum2D.xMax(), frustum2D.yMin(), 1);
    rayDirections[3] = sampleToCamera * vector3(frustum2D.xMin(), frustum2D.yMin(), 1);

#if ENABLE_HACKY_WIDE_FOV
    const float invWidth = 1.0f / 2160.0f;
    const float invHeight = 1.0f / 1200.0f;
    // TODO: undo sample-space padding of tile extents, and calculate correct padding in camera space
    float uv[4][2] = {
        frustum2D.xMin(), frustum2D.yMax(), frustum2D.xMax(), frustum2D.yMax(),
        frustum2D.xMax(), frustum2D.yMin(), frustum2D.xMin(), frustum2D.yMin(),
    };

    for (int i = 0; i < 4; i++) {
        float u = uv[i][0];
        float v = uv[i][1];

        float yaw = (u - .5f) * (HACKY_WIDE_FOV_W * RadiansPerDegree);
        float pitch = -(v - .5f) * (HACKY_WIDE_FOV_H * RadiansPerDegree);

        float newX = sin(yaw) * cos(pitch);
        float newY = sin(pitch);
        float newZ = -cos(yaw) * cos(pitch);
        rayDirections[i] = vector4(newX, newY, newZ);
    }

    return RayPacketFrustum3D(nearPoints[0], rayDirections[0], nearPoints[1], rayDirections[1], nearPoints[2],
                              rayDirections[2], nearPoints[3], rayDirections[3]);
#endif

    for (int i = 0; i < 4; ++i) {
        // printf("rayDirections[%d] = %s\n", i, toString(rayDirections[i]).c_str());
    }
    for (int i = 0; i < 4; ++i) {
        // printf("normalize(rayDirections[%d]) = %s\n", i, toString(normalize(rayDirections[i])).c_str());
    }

    // Compute extrema points on the focal plane
    vector3 focusPositions[4];
    for (int i = 0; i < 4; ++i) {
        focusPositions[i] = rayDirections[i] * (lens.focalDistance / -rayDirections[i].z);
    }

    for (int i = 0; i < 4; ++i) {
        // printf("focusPositions[%d] = %s\n", i, toString(focusPositions[i]).c_str());
    }

    // On the far plane
    float farXMax = -INFINITY;
    float farXMin = +INFINITY;
    float farYMax = -INFINITY;
    float farYMin = +INFINITY;
    // Construct all rays from extrema points on the lens AABB to extrema points on the far plane
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            vector3 rayPos = nearPoints[i];
            vector3 rayDir = normalize(focusPositions[j] - rayPos);
            // Intersect with far plane
            vector3 p = rayPos + rayDir * (farPlane / rayDir.z);
            farXMax = max(farXMax, p.x);
            farXMin = min(farXMin, p.x);
            farYMax = max(farYMax, p.y);
            farYMin = min(farYMin, p.y);
        }
    }
    vector3 farPoints[4];
    farPoints[0] = vector3(farXMin, farYMin, farPlane);
    farPoints[1] = vector3(farXMax, farYMin, farPlane);
    farPoints[2] = vector3(farXMax, farYMax, farPlane);
    farPoints[3] = vector3(farXMin, farYMax, farPlane);
    for (int i = 0; i < 4; ++i) {
        // printf("farPoints[%d] = %s\n", i, toString(farPoints[i]).c_str());
    }

    vector3 finalDirections[4];
    for (int i = 0; i < 4; ++i) {
        finalDirections[i] = normalize(farPoints[i] - nearPoints[i]);
    }
    for (int i = 0; i < 4; ++i) {
        // printf("finalDirections[%d] = %s\n", i, toString(finalDirections[i]).c_str());
    }

    return RayPacketFrustum3D(nearPoints[0], finalDirections[0], nearPoints[1], finalDirections[1], nearPoints[2],
                              finalDirections[2], nearPoints[3], finalDirections[3]);
}

// We'll take the easy way out for now and transform the 2D sample space rect into camera space.
// When switching to a general fit of non-pinhole camera space rays, we'll need to consider how
// the ray thickness (majorAxisLength) works in camera space (it's not a uniform thickness in
// camera space, unlike sample space).
void SampleHierarchy::populate3DFrom2D(uint32_t blockCount, const matrix3x3& sampleToCamera, ThinLens lens) {
    // TODO(mmara) set this on the camera itself?
    const float farPlane = -100.0f;

    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
        const auto& blockFrustum2D = blockFrusta2D[blockIndex];
        for (uint32_t tileIndex = 0; tileIndex < TILES_PER_BLOCK; ++tileIndex) {
            const auto& frustum2D = tileFrusta2D[blockIndex * TILES_PER_BLOCK + tileIndex];
            tileFrusta3D[blockIndex * TILES_PER_BLOCK + tileIndex] =
                get3DFrustumFrom2D(frustum2D, sampleToCamera, lens, farPlane);
        }
        blockFrusta3D[blockIndex] = get3DFrustumFrom2D(blockFrustum2D, sampleToCamera, lens, farPlane);
    }
}

void SampleHierarchy::generate(ArrayView<SortedSample> sortedSamples,
                               uint32_t blockCount,
                               uint32_t validSampleCount,
                               const FloatRect& cullRect,
                               ArrayView<float> blockedSamplePositions,
                               ArrayView<Sample::Extents> blockedSampleExtents,
                               ThinLens lens,
                               const matrix3x3& sampleToCamera) {
    uint32_t maxIndex = validSampleCount - 1;
    uint32_t sampleIndex = 0;
    RayPacketFrustum2D cullFrustum2D(cullRect.lower.x, cullRect.upper.x, cullRect.lower.y, cullRect.upper.y);
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
        auto& blockFrustum2D = blockFrusta2D[blockIndex];
        blockFrustum2D.setEmpty();
        for (uint32_t tileIndex = 0; tileIndex < TILES_PER_BLOCK; ++tileIndex) {
            auto& frustum2D = tileFrusta2D[blockIndex * TILES_PER_BLOCK + tileIndex];
            frustum2D.setEmpty();
            for (uint32_t tileSample = 0; tileSample < TILE_SIZE; tileSample++) {
                float x = sortedSamples[sampleIndex].position.x;
                float y = sortedSamples[sampleIndex].position.y;
                float major = sortedSamples[sampleIndex].extents.majorAxisLength;
                frustum2D.merge(x + major, y + major);
                frustum2D.merge(x - major, y - major);

                blockedSamplePositions[sampleIndex * 2] = x;
                blockedSamplePositions[sampleIndex * 2 + 1] = y;
                blockedSampleExtents[sampleIndex] = sortedSamples[sampleIndex].extents;

                // Copy the final sample to pad out the block
                sampleIndex = std::min(sampleIndex + 1, maxIndex);
            }
            frustum2D.intersect(cullFrustum2D);
            blockFrustum2D.merge(frustum2D);
        }
    }

    populate3DFrom2D(blockCount, sampleToCamera, lens);
}

} // namespace hvvr
