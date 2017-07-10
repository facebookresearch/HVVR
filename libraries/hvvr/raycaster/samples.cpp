/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "samples.h"
#include "constants_math.h"

#include <assert.h>
#include <cmath>
#include <functional>
#include <random>
#include <vector>


namespace hvvr {

DynamicArray<Sample> getGridSamples(size_t width, size_t height) {
    float invWidth = 1.0f / width;
    float invHeight = 1.0f / height;
    DynamicArray<Sample> samples(width * height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t index = y * width + x;
            samples[index].pixelLocation.x = (uint32_t)x;
            samples[index].pixelLocation.y = (uint32_t)y;
            samples[index].position.x = ((float)x + 0.5f) * invWidth;
            samples[index].position.y = ((float)y + 0.5f) * invHeight;
            if (width > height) {
                samples[index].extents.minorAxis = {invWidth, 0};
                samples[index].extents.majorAxisLength = invHeight;
            } else {
                samples[index].extents.minorAxis = {0, invHeight};
                samples[index].extents.majorAxisLength = invWidth;
            }
        }
    }
    return samples;
}

DynamicArray<DirectionSample> getEyeSpacePolarFoveatedSamples(std::vector<float>& ringEccentricities,
                                                                    size_t& samplesPerRing,
                                                                    float maxEyeTrackingUncertaintyDegrees,
                                                                    float minMAR,
                                                                    float maxMAR,
                                                                    float maxFOVDegrees,
                                                                    float marSlope,
                                                                    float fovealMARDegrees,
                                                                    float zenithJitterStrength,
                                                                    float ringJitterStrength) {
    assert(zenithJitterStrength <= 1.0f);
    assert(ringJitterStrength <= 1.0f);
    const float m = marSlope;
    const float w_0 = fovealMARDegrees;

    samplesPerRing = 0;
    int ringCount = 0;
    size_t irregularGridSampleCount = 0;
    { // Calculate number of samples so we can allocate before generation
        // w = m e + w_0
        // e = (w - w_0 / m)
        float e = (minMAR - w_0) / m;
        float w = w_0;
        while (e - w <= maxFOVDegrees) {
            // Angular distance (in degrees) between samples on this annulus
            w = std::min(maxMAR, m * std::max(e - maxEyeTrackingUncertaintyDegrees, 0.0f) + w_0);
            float ringRadius = sinf(e * RadiansPerDegree);
            float angularDistanceAroundRing = 2.0f * Pi * ringRadius;
            float angularDistanceAroundRingDegrees = angularDistanceAroundRing / RadiansPerDegree;
            size_t samplesOnAnnulus = (size_t)(std::ceil(angularDistanceAroundRingDegrees / w));
            printf("New samplesOnAnnulus: %zu = ceil(%f/%f)\n", samplesOnAnnulus, angularDistanceAroundRingDegrees, w);
            samplesPerRing = std::max(samplesPerRing, samplesOnAnnulus);

            irregularGridSampleCount += samplesOnAnnulus;
            e += w;
            ++ringCount;
        }
    }
    printf("(%zu*%d=%d)/%d %f times the minimal sample count\n", samplesPerRing, ringCount,
           (int)(samplesPerRing * ringCount), (int)irregularGridSampleCount,
           (samplesPerRing * ringCount) / (float)irregularGridSampleCount);

    int index = 0;
    DynamicArray<DirectionSample> samples(ringCount * samplesPerRing);
    float e = (minMAR - w_0) / m;
    float w = w_0;
    /**
    A note on differentials:
    The zenith differential is found by taking the nearest point on the next concentric ring outwards, and
    projecting it onto
    the tangent plane of the current direction (where the direction is on the unit sphere)
    The vector from the intersection point of the current direction and the tangent plane to the intersection point
    of the
    "nearest direction on the next ring" and the tangent plane is the zenith differential

    Ray-plane intersection:
    Ray: P = P_0 + tV
    Plane: dot(P, N) + d = 0
    For our purposes, d = 1, P_0 is zero, so we actually have
    Ray: P = tV
    Plane: dot(P, N) = -1
    We'll reverse the convention, so that the normal of the plane faces away from the origin
    Ray: P = tV
    Plane: dot(P, N) = 1
    dot(tV,N) = 1
    t * dot(V,N) = 1
    t = 1 / dot(V,N)
    P = t*V
    P = V/dot(V,N) <- all we need

    Then the zenith differential is just P-N

    The azimuthal differential is taken by finding a perpendicular vector to the zenith differential,
    and then scaling it to the distance along the ring to the next sample
    */
    // Generate concentric circles of samples with spacing equal or less than MAR of eye at the eccentricity

    std::uniform_real_distribution<float> uniformRandomDist(0.0f, 1.0f);
    std::mt19937 generator;
    auto rand = std::bind(uniformRandomDist, std::ref(generator));

    while (e - w <= maxFOVDegrees) {
        // Angular distance (in degrees) between samples on this annulus
        w = std::min(maxMAR, m * std::max(e - maxEyeTrackingUncertaintyDegrees, 0.0f) + w_0);
        float ringRadius = sinf(e * RadiansPerDegree);
        float angularDistanceAroundRing = 2.0f * Pi * ringRadius;

        ringEccentricities.push_back(e * RadiansPerDegree);
        float ringRotation = (rand() - 0.5f) * ringJitterStrength;
        for (int i = 0; i < samplesPerRing; ++i) {
            float zenithJitter = w * (rand() - 0.5f) * zenithJitterStrength * 0.5f;
            vector3 baseVector = normalize(
                quaternion::fromAxisAngle(vector3(0, 1, 0), (e + zenithJitter) * RadiansPerDegree) * vector3(0, 0, -1));
            vector3 zenithDiffBaseVector =
                normalize(quaternion::fromAxisAngle(vector3(0, 1, 0), (e + w + zenithJitter) * RadiansPerDegree) *
                          vector3(0, 0, -1));

            float rotationRadians = (i + ringRotation + 0.5f) / float(samplesPerRing) * 2.0f * Pi;
            vector3 p = normalize(quaternion::fromAxisAngle(vector3(0, 0, -1), rotationRadians) * baseVector);
            vector3 zenithDiffDirection =
                normalize(quaternion::fromAxisAngle(vector3(0, 0, -1), rotationRadians) * zenithDiffBaseVector);

            // Project zenith direction onto tangent plane of p
            // P = V/dot(V,N)
            vector3 zenithDiffDirectionOnTangentPlane = zenithDiffDirection * (1.0f / dot(zenithDiffDirection, p));
            vector3 zenithDiff = zenithDiffDirectionOnTangentPlane - p;
            vector3 azimuthalDiffUnit = cross(p, normalize(zenithDiff));
            vector3 azimuthalDiff = azimuthalDiffUnit * angularDistanceAroundRing / (float)samplesPerRing;
            samples[index].direction = p;
            samples[index].zenithDifferential = zenithDiff;
            samples[index].azimuthalDifferential = azimuthalDiff;
            ++index;
        }

        e += w;
    }
    return samples;
}

} // namespace hvvr
