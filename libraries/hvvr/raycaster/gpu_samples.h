#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "constants_math.h"
#include "cuda_decl.h"
#include "kernel_constants.h"
#include "samples.h"
#include "vector_math.h"


namespace hvvr {

struct GPUCamera;

// From http://graphics.cs.williams.edu/papers/DeepGBuffer16/Mara2016DeepGBuffer.pdf
static constexpr int tau[] = {
    //  0   1   2   3   4   5   6   7   8   9
    1,  1,  1,  2,  3,  2,  5,  2,  3,  2,   // 0
    3,  3,  5,  5,  3,  4,  7,  5,  5,  7,   // 1
    9,  8,  5,  5,  7,  7,  7,  8,  5,  8,   // 2
    11, 12, 7,  10, 13, 8,  11, 8,  7,  14,  // 3
    11, 11, 13, 12, 13, 19, 17, 13, 11, 18,  // 4
    19, 11, 11, 14, 17, 21, 15, 16, 17, 18,  // 5
    13, 17, 11, 17, 19, 18, 25, 18, 19, 19,  // 6
    29, 21, 19, 27, 31, 29, 21, 18, 17, 29,  // 7
    31, 31, 23, 18, 25, 26, 25, 23, 19, 34,  // 8
    19, 27, 21, 25, 39, 29, 17, 21, 27, 29}; // 9
template <uint32_t MSAARate>
CUDA_HOST_DEVICE_INL vector2 tapLocation(int subsampleIndex, float spinAngle, float startOffset, float& radius) {
    const int numSpiralTurns = tau[MSAARate];
    // Radius relative to ssR
    float alpha = float(subsampleIndex + startOffset) * (1.0f / MSAARate);
    float angle = alpha * (numSpiralTurns * 6.28f) + spinAngle;

    radius = sqrtf(alpha);
    return vector2(cosf(angle), sinf(angle));
}

template <uint32_t MSAARate>
CUDA_HOST_DEVICE_INL vector2 getSubsampleUnitOffset(vector2 sampleJitter,
                                                   int subsampleIndex,
                                                   float extraSpinAngle = 0.0f) {
    (void)sampleJitter;
    float spinAngle = extraSpinAngle;
#if JITTER_SAMPLES
    spinAngle += sampleJitter.x * 6.28f;
#endif
    float startOffset = 0.5f;

    float radius;
    vector2 unitDiskLoc = tapLocation<MSAARate>(subsampleIndex, spinAngle, startOffset, radius);

    return vector2(unitDiskLoc.x * radius, unitDiskLoc.y * radius);
}

struct SampleInfo {
    vector2* centers;
    Sample::Extents* extents;
    vector2 frameJitter;
    ThinLens lens;
    SampleInfo(const GPUCamera& camera);
};

struct UnpackedSample {
    vector2 center;
    vector2 majorAxis;
    vector2 minorAxis;
};

struct UnpackedDirectionalSample {
    vector3 centerDir;
    vector3 majorDirDiff;
    vector3 minorDirDiff;
};

struct SampleDoF {
    vector3 pos;
    vector3 dir;
};

// sqrt(2)/2, currently a hack so that the ellipses blobs of diagonally adjacent pixels on a uniform grid are tangent
#define EXTENT_MODIFIER 0.70710678118f

CUDA_DEVICE_INL UnpackedSample GetFullSample(uint32_t sampleIndex, SampleInfo sampleInfo) {
    UnpackedSample sample;
    sample.center = sampleInfo.centers[sampleIndex];

    Sample::Extents extents = sampleInfo.extents[sampleIndex];
    sample.minorAxis.x = extents.minorAxis.x * EXTENT_MODIFIER;
    sample.minorAxis.y = extents.minorAxis.y * EXTENT_MODIFIER;

    // 90 degree Rotation, and rescale
    float minorAxisLengthInv =
        rsqrtf(sample.minorAxis.x * sample.minorAxis.x + sample.minorAxis.y * sample.minorAxis.y);
    float rescale = extents.majorAxisLength * EXTENT_MODIFIER * minorAxisLengthInv;
    sample.majorAxis.x = -sample.minorAxis.y * rescale;
    sample.majorAxis.y = sample.minorAxis.x * rescale;

    return sample;
}

CUDA_DEVICE_INL UnpackedDirectionalSample GetDirectionalSample3D(uint32_t sampleIndex,
                                                                 SampleInfo sampleInfo,
                                                                 matrix4x4 sampleToWorld,
                                                                 matrix3x3 sampleToCamera,
                                                                 matrix4x4 cameraToWorld) {
    UnpackedSample sample = GetFullSample(sampleIndex, sampleInfo);

#if ENABLE_HACKY_WIDE_FOV
    matrix3x3 cameraToWorldRotation = matrix3x3(cameraToWorld);

    UnpackedDirectionalSample sample3D;

    float u = sample.center.x;
    float v = sample.center.y;

    float yaw = (u - .5f) * (HACKY_WIDE_FOV_W * RadiansPerDegree);
    float pitch = -(v - .5f) * (HACKY_WIDE_FOV_H * RadiansPerDegree);

    float newX = sin(yaw) * cos(pitch);
    float newY = sin(pitch);
    float newZ = -cos(yaw) * cos(pitch);
    sample3D.centerDir = vector3(newX, newY, newZ);

    // making something up...
    const float invWidth = 1.0f / 2160.0f;
    const float invHeight = 1.0f / 1200.0f;
    float majorAxisMag = sin(.5f * invHeight * (HACKY_WIDE_FOV_H * RadiansPerDegree));
    float minorAxisMag = sin(.5f * invWidth * (HACKY_WIDE_FOV_W * RadiansPerDegree));

    sample3D.majorDirDiff.x = sin(yaw) * sin(pitch);
    sample3D.majorDirDiff.y = -cos(pitch);
    sample3D.majorDirDiff.z = -cos(yaw) * sin(pitch);

    sample3D.minorDirDiff = cross(sample3D.majorDirDiff, sample3D.centerDir);

    sample3D.majorDirDiff *= majorAxisMag;
    sample3D.minorDirDiff *= minorAxisMag;

    if (HACKY_WIDE_FOV_H > HACKY_WIDE_FOV_W) {
        vector3 temp = sample3D.minorDirDiff;
        sample3D.minorDirDiff = sample3D.majorDirDiff;
        sample3D.majorDirDiff = temp;
    }

    sample3D.centerDir = cameraToWorldRotation * sample3D.centerDir;
    sample3D.majorDirDiff = cameraToWorldRotation * sample3D.majorDirDiff;
    sample3D.minorDirDiff = cameraToWorldRotation * sample3D.minorDirDiff;
#else
    matrix3x3 sampleToWorldRotation(sampleToWorld);

    UnpackedDirectionalSample sample3D;
    sample3D.centerDir = sampleToWorldRotation * vector3(sample.center.x, sample.center.y, 1.0f);
    sample3D.majorDirDiff = sampleToWorldRotation * vector3(sample.majorAxis.x, sample.majorAxis.y, 0.0f);
    sample3D.minorDirDiff = sampleToWorldRotation * vector3(sample.minorAxis.x, sample.minorAxis.y, 0.0f);
#endif

    return sample3D;
}

template <uint32_t MSAARate, uint32_t BlockSize>
CUDA_DEVICE_INL void GetSampleUVsDoF(const vector2* CUDA_RESTRICT tileSubsampleLensPos,
                                     vector2 frameJitter,
                                     vector2 focalToLensScale,
                                     int subsample,
                                     vector2& lensUV,
                                     vector2& dirUV) {
    // Random position on lens. As lensRadius approaches zero, depth of field rays become equivalent to
    // non-depth of field rays, including AA subsample pattern.
    int lensPosIndex =
        (blockIdx.x % DOF_LENS_POS_LOOKUP_TABLE_TILES) * BlockSize * MSAARate + subsample * BlockSize + threadIdx.x;
    // 1 LDG.64 (coherent, but half-efficiency)
    lensUV = tileSubsampleLensPos[lensPosIndex];

    // compile-time constant
    vector2 aaOffset = getSubsampleUnitOffset<MSAARate>(frameJitter, subsample);

    // 2 FMA
    dirUV = aaOffset * focalToLensScale - lensUV;
}

} // namespace hvvr
