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


namespace hvvr {

struct TileData {
    vector3 rayOrigin;
    vector3 majorDirDiff;
    vector3 minorDirDiff;

    CUDA_DEVICE void load(matrix4x4 sampleToWorld, UnpackedDirectionalSample sample) {
        rayOrigin = vector3(sampleToWorld * vector4(0.0f, 0.0f, 0.0f, 1.0f));

        majorDirDiff = sample.majorDirDiff;
        minorDirDiff = sample.minorDirDiff;
    }
};

struct TileDataDoF {
    vector2 focalToLensScale;

    vector3 lensCenter;
    vector3 lensU;
    vector3 lensV;

    CUDA_DEVICE void load(SampleInfo sampleInfo, matrix4x4 sampleToWorld, uint32_t sampleOffset) {
        UnpackedSample sample2D = GetFullSample(sampleOffset, sampleInfo);
        matrix3x3 sampleToWorldRotation = matrix3x3(sampleToWorld);

        lensCenter = vector3(sampleToWorld * vector4(0.0f, 0.0f, 0.0f, 1.0f));

        // actual focal derivatives should be multiplier by sampleInfo.lens.focalDistance (see scale below)
        vector3 focalU = sampleToWorldRotation * vector3(sample2D.majorAxis.x, sample2D.majorAxis.y, 0.0f);
        vector3 focalV = sampleToWorldRotation * vector3(sample2D.minorAxis.x, sample2D.minorAxis.y, 0.0f);

        // If we force the lens and focal planes to be parallel, and their derivatives to be identical except for a
        // scale factor, we can optimize the inner loop and test.
        float focalDistance = sampleInfo.lens.focalDistance;
        float lensRadius = sampleInfo.lens.radius;
        float focalUMagInv = rsqrtf(dot(focalU, focalU));
        float focalVMagInv = rsqrtf(dot(focalV, focalV));
        float focalUMag = 1.0f / focalUMagInv;
        float focalVMag = 1.0f / focalVMagInv;
        lensU = focalU * (lensRadius * focalUMagInv);
        lensV = focalV * (lensRadius * focalVMagInv);
        focalToLensScale = (focalDistance / lensRadius) * vector2(focalUMag, focalVMag);
    }
};

} // namespace hvvr
