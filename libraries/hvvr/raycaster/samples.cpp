/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "samples.h"
#include "constants_math.h"
#include "foveated.h"
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

DynamicArray<DirectionalBeam> getEyeSpacePolarFoveatedSamples(size_t& samplesPerRing,
                                                              EccentricityMap& emap,
                                                              float maxEyeTrackingUncertaintyDegrees,
                                                              float maxFOVDegrees,
                                                              float marSlope,
                                                              float fovealMARDegrees) {
    const float m = marSlope;
    const float w_0 = fovealMARDegrees;
    const float switchPoint1 = maxEyeTrackingUncertaintyDegrees / w_0;
    const float S = maxEyeTrackingUncertaintyDegrees;
    emap = EccentricityMap(marSlope, fovealMARDegrees, maxEyeTrackingUncertaintyDegrees);

    samplesPerRing = 0;
    size_t ringCount = 0;
    size_t irregularGridSampleCount = 0;
    { // Calculate number of samples so we can allocate before generation
        float E = 0.0f;
        while (E <= maxFOVDegrees * RadiansPerDegree) {
            // Angular distance (in degrees) between samples on this annulus
            float w = emap.apply(ringCount + 1.0f) - E;
            float ringRadius = sinf(E);
            float angularDistanceAroundRing = 2.0f * Pi * ringRadius;
            size_t samplesOnAnnulus = (size_t)(std::ceil(angularDistanceAroundRing / w));
            printf("New samplesOnAnnulus: %zu = ceil(%f/%f)\n", samplesOnAnnulus, angularDistanceAroundRing, w);
            samplesPerRing = std::max(samplesPerRing, samplesOnAnnulus);

            irregularGridSampleCount += samplesOnAnnulus;
            ++ringCount;
            E = emap.apply((float)ringCount);
        }
    }
    printf("(%zu*%zu=%d)/%d %f times the minimal sample count\n", samplesPerRing, ringCount,
           (int)(samplesPerRing * ringCount), (int)irregularGridSampleCount,
           (samplesPerRing * ringCount) / (float)irregularGridSampleCount);
    DynamicArray<DirectionalBeam> samples(ringCount * samplesPerRing);
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
    float E = 0.0f;
    for (size_t r = 0; r < ringCount; ++r) {
        // Angular distance (in degrees) between samples on this annulus
        float E_next = emap.apply(r + 1.0f);
        float ringRadius = sinf(E);
        float angularDistanceAroundRing = 2.0f * Pi * ringRadius;

        vector3 baseVector = normalize(quaternion::fromAxisAngle(vector3(0, 1, 0), E) * vector3(0, 0, -1));
        vector3 zenithDiffBaseVector =
            normalize(quaternion::fromAxisAngle(vector3(0, 1, 0), E_next) * vector3(0, 0, -1));

        for (size_t i = 0; i < samplesPerRing; ++i) {
            float rotationRadians = (i + 0.5f) / float(samplesPerRing) * 2.0f * Pi;
            vector3 p = normalize(quaternion::fromAxisAngle(vector3(0, 0, -1), rotationRadians) * baseVector);
            vector3 zenithDiffDirection =
                normalize(quaternion::fromAxisAngle(vector3(0, 0, -1), rotationRadians) * zenithDiffBaseVector);

            // Project zenith direction onto tangent plane of p
            // P = V/dot(V,N)
            vector3 zenithDiffDirectionOnTangentPlane = zenithDiffDirection * (1.0f / dot(zenithDiffDirection, p));
            vector3 zenithDiff = zenithDiffDirectionOnTangentPlane - p;
            vector3 azimuthalDiffUnit = cross(p, normalize(zenithDiff));
            vector3 azimuthalDiff = azimuthalDiffUnit * angularDistanceAroundRing / (float)samplesPerRing;
            size_t idx = samplesPerRing * r + i;
            samples[idx].centerRay = p;
            samples[idx].du = zenithDiff;
            samples[idx].dv = azimuthalDiff;
        }

        E = E_next;
    }

    return samples;
}

UnpackedSample unpackSample(Sample s) {
    UnpackedSample sample;
    // sqrt(2)/2, currently a hack so that the ellipses blobs of diagonally adjacent pixels on a uniform grid are
    // tangent
#define EXTENT_MODIFIER 0.70710678118f
    sample.center = s.position;
    sample.minorAxis.x = s.extents.minorAxis.x * EXTENT_MODIFIER;
    sample.minorAxis.y = s.extents.minorAxis.y * EXTENT_MODIFIER;

    // 90 degree Rotation, and rescale
    float rescale = s.extents.majorAxisLength * EXTENT_MODIFIER / length(s.extents.minorAxis);
    sample.majorAxis.x = -sample.minorAxis.y * rescale;
    sample.majorAxis.y = sample.minorAxis.x * rescale;
    return sample;
#undef EXTENT_MODIFIER
}

void saveSamples(const std::vector<hvvr::Sample>& samples, const std::string& filename) {
    auto file = fopen(filename.c_str(), "wb");
    if (!file) {
        hvvr::fail("Unable to open output sample file %s", filename.c_str());
    }
    SampleFileHeader header;
    header.sampleCount = uint32_t(samples.size());
    fwrite(&header, sizeof(SampleFileHeader), 1, file);
    fwrite(&samples[0], sizeof(hvvr::Sample), header.sampleCount, file);
    fclose(file);
}

void loadSamples(hvvr::DynamicArray<hvvr::Sample>& samples, const std::string& filename) {
    auto file = fopen(filename.c_str(), "rb");
    if (!file) {
        hvvr::fail("Unable to find sample file %s\nMake sure to generate them using GenerateSamplesFromDistortion "
                   "then copy them to this project's folder",
                   filename.c_str());
    }
    SampleFileHeader header;
    fread(&header, sizeof(SampleFileHeader), 1, file);
    assert(header.magic == SampleFileHeader().magic);
    assert(header.version == SampleFileHeader().version);
    samples = hvvr::DynamicArray<hvvr::Sample>(header.sampleCount);
    fread(samples.data(), sizeof(hvvr::Sample), samples.size(), file);
    fclose(file);
}


} // namespace hvvr
