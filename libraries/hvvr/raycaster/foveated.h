#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "graphics_types.h"
#include "raycaster_spec.h"
#include "sample_hierarchy.h"
#include <constants_math.h>
#include <cuda_decl.h>

namespace hvvr {

class Raycaster;

struct FoveatedSampleData {
    size_t blockCount = 0;
    BeamBatch samples;
    DynamicArray<SimpleRayFrustum> simpleTileFrusta;
    DynamicArray<SimpleRayFrustum> simpleBlockFrusta;
};

/* Abstraction of the mapping R -> E where:
R is the space of ring coordinates and
E is the eccentricity, given in degrees
*/
struct EccentricityMap {
    float m;
    float w_0;
    float S;

    // Precomputed potentially expensive intermediates
    // log(m+1)
    float invLogA;
    // 1 / w_0
    float invW_0;
    // 1 / m
    float invM;

    EccentricityMap() {}
    EccentricityMap(float marSlope, float maxMARDegrees, float maxResolutionDegrees) {
        m = marSlope;
        w_0 = maxMARDegrees * RadiansPerDegree;
        S = maxResolutionDegrees * RadiansPerDegree;

        // Compute in double precision since it only happens once
        invLogA = float(1.0 / log(1.0 + (double)m));
        invW_0 = float(1.0 / w_0);
        invM = float(1.0 / m);
    }
    /**
    Mapping from ring coordinates to eccentricity, where E(0) = 0, where we don't start
    the resolution falloff until the eccentricity is S
    let S = maxEyeTrackingUncertaintyDegrees

    This is a function that is a continuous generalization of this recurrence:
    w(n) = m * max(E(n) - S, 0) + w_0
    E(n+1) = E(n) + w(n)

    The character of the function is obviously different starting at E(n) - S = 0, so we can
    split it piecewise
    w(n) = w_0              | E(n) <= S
    w(n) = m*(E(n)-S) + w_0 | E(n) > S

    For  E(n) <= S, this results in the linear equation:
    E(n) = w_0*n

    Eccentricity intially increases at a constant rate (within the radius of uncertainty).

    Now for the more complicated case

    Solving a recurrence relation for the hyperbolic falloff (after a simple coordinate transform):
    let x = (w_0*n-S) / w_0
    let a = 1+m
    Then we are solving for g(x) where:
    g(x+1) = a*g(x)+w_0
    and
    g(0) = 0
    Which gives us (https://m.wolframalpha.com/input/?i=g%28n%2B1%29%3Da*g%28n%29%2Bw%2C+g%280%29%3D0):
    g(x) = (w_0*(a^x-1))/(a-1)

    Then we can transform the coordinates back:
    E(n) = g((w_0*n-S) / w_0)+S

    We can invert g to help get a map from eccentricity to ring location:
    g = (w_0*(a^x-1))/(a-1)
    (a-1)g/w_0=(a^x-1)
    a^x=(a-1)g/w_0+1
    x = ln((a-1)g/w_0+1)/ln(a)

    */
    // Transform ring coordinates to eccentricity, given in radians
    CHD float apply(float i) {
        float e0 = i * w_0;
        float x = (e0 - S) * invW_0;
        float g_x = (w_0 * (powf(1.0f + m, x) - 1.0f)) * invM;
        float e1 = g_x + S;
        return (e0 < S) ? e0 : e1;
    };
    // Transform eccentricity, given in radians, to ring coordinates
    CHD float applyInverse(float E) {
        float i0 = E * invW_0;
        float sDivW = S * invW_0;
        float x = logf((m * (i0 - sDivW)) + 1.0f) * invLogA;
        float i1 = x + sDivW;
        return (E < S) ? i0 : i1;
    };
};


// TODO(anankervis): merge into Raycaster class
void generateEyeSpacePolarFoveatedSampleData(FoveatedSampleData& foveatedSampleData,
                                             std::vector<vector2ui>& polarRemapToPixel,
                                             EccentricityMap& eccentricityMap,
                                             size_t& samplesPerRing,
                                             RayCasterSpecification::FoveatedSamplePattern pattern);

void polarSpaceFoveatedSetup(Raycaster* raycaster);

} // namespace hvvr
