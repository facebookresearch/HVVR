/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sample_hierarchy.h"
#include "constants_math.h"
#include "kernel_constants.h"

#include <sstream>

#pragma warning(disable : 4505) // unreferenced local function has been removed


namespace hvvr {

static std::string toString(vector4 v) {
    std::ostringstream stringStream;
    stringStream << "vector4(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return stringStream.str();
}


static vector3 sphericalUVToDirection(vector2 uv, float fovX, float fovY) {
    float yaw = (uv.x - .5f) * (fovX * RadiansPerDegree);
    float pitch = -(uv.y - .5f) * (fovY * RadiansPerDegree);

    float newX = sin(yaw) * cos(pitch);
    float newY = sin(pitch);
    float newZ = -cos(yaw) * cos(pitch);
    return normalize(vector3(newX, newY, newZ));
}

// http://cseweb.ucsd.edu/~ravir/whitted.pdf outlines a basic technique for generating bounding frusta over a packet of
// non-point origin rays Step 1: Pick a major axis for the rays Step 2: Choose a near and far plane for the rays,
// perpendicular to the major axis Step 3: Compute AABB of the ray intersection points on both of the planes Step 4:
// Connect the corresponding corners of the AABBs with rays, these 4 rays define the corner rays of a conservative
// bounding frustum

// When converting from 2D samples to 3D, we can simplify this a bit
// Choose the major axis to be the Z axis. The near plane is z=0, the far plane can just be a camera parameter (negative
// Z) The AABB for the near plane is just the AABB of the lens.
static Frustum get3DFrustumFrom2D(const RayPacketFrustum2D& frustum2D,
                                             Sample2Dto3DMappingSettings settings) {
    auto lens = settings.thinLens;
    auto sampleToCamera = settings.sampleToCamera;

    const float farPlane = -100.0f;

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

    if (settings.type == Sample2Dto3DMappingSettings::MappingType::SphericalSection) {
        // TODO: undo sample-space padding of tile extents, and calculate correct padding in camera space
        float uv[4][2] = {
            frustum2D.xMin(), frustum2D.yMax(), frustum2D.xMax(), frustum2D.yMax(),
            frustum2D.xMax(), frustum2D.yMin(), frustum2D.xMin(), frustum2D.yMin(),
        };

        for (int i = 0; i < 4; i++) {
            vector2 uvCurrent = {uv[i][0], uv[i][1]};
            rayDirections[i] = sphericalUVToDirection(uvCurrent, settings.fovXDegrees, settings.fovYDegrees);
        }

        return Frustum(nearPoints, rayDirections);
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

    vector3 finalDirections[4];
    for (int i = 0; i < 4; ++i) {
        finalDirections[i] = normalize(farPoints[i] - nearPoints[i]);
    }

    return Frustum(nearPoints, finalDirections);
}

// We'll take the easy way out for now and transform the 2D sample space rect into camera space.
// When switching to a general fit of non-pinhole camera space rays, we'll need to consider how
// the ray thickness (majorAxisLength) works in camera space (it's not a uniform thickness in
// camera space, unlike sample space).
void BeamBatch::generateFrom2D(const BeamBatch2D& hierarchy2D, Sample2Dto3DMappingSettings settings) {
    for (uint32_t blockIndex = 0; blockIndex < hierarchy2D.blockFrusta.size(); ++blockIndex) {
        const auto& blockFrustum2D = hierarchy2D.blockFrusta[blockIndex];
        for (uint32_t tileIndex = 0; tileIndex < TILES_PER_BLOCK; ++tileIndex) {
            const auto& frustum2D = hierarchy2D.tileFrusta[blockIndex * TILES_PER_BLOCK + tileIndex];
            tileFrusta3D[blockIndex * TILES_PER_BLOCK + tileIndex] = get3DFrustumFrom2D(frustum2D, settings);
        }
        blockFrusta3D[blockIndex] = get3DFrustumFrom2D(blockFrustum2D, settings);
    }
    for (uint32_t sampleIndex = 0; sampleIndex < hierarchy2D.samples.size(); ++sampleIndex) {
        UnpackedSample us = hierarchy2D.samples[sampleIndex];
        DirectionalBeam& ds = directionalBeams[sampleIndex];
        ds.centerRay = settings.sampleToCamera * vector3(us.center, 1.0f);
        ds.du = settings.sampleToCamera * vector3(us.majorAxis, 0.0f);
        ds.dv = settings.sampleToCamera * vector3(us.minorAxis, 0.0f);
        if (settings.type == Sample2Dto3DMappingSettings::MappingType::SphericalSection) {
            ds.centerRay = sphericalUVToDirection(us.center, settings.fovXDegrees, settings.fovYDegrees);
            ds.du = sphericalUVToDirection(us.center + us.majorAxis, settings.fovXDegrees, settings.fovYDegrees) -
                    ds.centerRay;
            ds.dv = sphericalUVToDirection(us.center + us.minorAxis, settings.fovXDegrees, settings.fovYDegrees) -
                    ds.centerRay;
        }
    }
}

BeamBatch2D::BeamBatch2D(ArrayView<SortedSample> sortedSamples,
                         uint32_t blockCount,
                         uint32_t validSampleCount,
                         const FloatRect& cullRect,
                         ThinLens lens,
                         const matrix3x3& sampleToCamera) {
    (void)lens;
    (void)sampleToCamera;
    uint32_t maxIndex = validSampleCount - 1;
    uint32_t sampleIndex = 0;
    RayPacketFrustum2D cullFrustum2D(cullRect.lower.x, cullRect.upper.x, cullRect.lower.y, cullRect.upper.y);
    blockFrusta = DynamicArray<RayPacketFrustum2D>(blockCount);
    tileFrusta = DynamicArray<RayPacketFrustum2D>(blockCount * TILES_PER_BLOCK);
    samples = DynamicArray<UnpackedSample>(blockCount * BLOCK_SIZE);
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
        auto& blockFrustum2D = blockFrusta[blockIndex];
        blockFrustum2D.setEmpty();
        for (uint32_t tileIndex = 0; tileIndex < TILES_PER_BLOCK; ++tileIndex) {
            auto& frustum2D = tileFrusta[blockIndex * TILES_PER_BLOCK + tileIndex];
            frustum2D.setEmpty();
            for (uint32_t tileSample = 0; tileSample < TILE_SIZE; tileSample++) {
                auto s = sortedSamples[sampleIndex];
                float x = s.position.x;
                float y = s.position.y;
                float major = s.extents.majorAxisLength;
                frustum2D.merge(x + major, y + major);
                frustum2D.merge(x - major, y - major);

                samples[sampleIndex] = unpackSample(sortedSamples[sampleIndex]);
                // Copy the final sample to pad out the block
                sampleIndex = std::min(sampleIndex + 1, maxIndex);
            }
            frustum2D.intersect(cullFrustum2D);
            blockFrustum2D.merge(frustum2D);
        }
    }
}

} // namespace hvvr
