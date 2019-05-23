#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gpu_samples.h"


namespace hvvr {

template <bool DoF>
struct TileData;


template <>
struct TileData<false> {
    vector3 rayOrigin;
    vector3 dirDu;
    vector3 dirDv;

    CUDA_DEVICE_INL void load(matrix4x4 cameraToWorld, DirectionalBeam sample) {
        rayOrigin = vector3(cameraToWorld.m3);
        dirDu = sample.du;
        dirDv = sample.dv;
    }
};

template <>
struct TileData<true> {
    vector2 focalToLensScale;
    vector3 lensU;
    vector3 lensV;

    CUDA_DEVICE void load(CameraBeams cameraBeams, matrix4x4 cameraToWorld, DirectionalBeam sample) {
        matrix3x3 cameraToWorldRotation = matrix3x3(cameraToWorld);

        // actual focal derivatives should be multiplier by cameraBeams.lens.focalDistance (see scale below)
        vector3 focalU =
            cameraToWorldRotation * sample.du; // vector3(sample2D.majorAxis.x, sample2D.majorAxis.y, 0.0f);
        vector3 focalV =
            cameraToWorldRotation * sample.dv; // vector3(sample2D.minorAxis.x, sample2D.minorAxis.y, 0.0f);

        // If we force the lens and focal planes to be parallel, and their derivatives to be identical except for a
        // scale factor, we can optimize the inner loop and test.
        float focalDistance = cameraBeams.lens.focalDistance;
        float lensRadius = cameraBeams.lens.radius;
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
