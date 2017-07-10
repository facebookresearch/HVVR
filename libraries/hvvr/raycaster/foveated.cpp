/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "foveated.h"
#include "camera.h"
#include "constants_math.h"
#include "cuda_raycaster.h"
#include "raycaster.h"
#include "sample_clustering.h"


namespace hvvr {

void generateRemapForFoveatedSamples(ArrayView<DirectionSample> unsortedSamples,
                                     ArrayView<uint32_t> remap) { // Create a remapping from the original directional
                                                                  // samples to ones that are binned decently

    struct DirectionalSampleToSort {
        vector2 position; // eccentricity,theta (octahedral is a bad match for the naive clustering algorithm)
        uint32_t originalIndex;
    };
    DynamicArray<DirectionalSampleToSort> toSort(unsortedSamples.size());
    for (size_t i = 0; i < toSort.size(); ++i) {
        vector3 v = unsortedSamples[i].direction;
        // Eccentricity angle
        toSort[i].position.x = acosf(-v.z);
        vector3 dir = vector3(v.x, v.y, 0.0f);
        // Angle of rotation about z, measured from x
        toSort[i].position.y = atan2f(normalize(dir).y, normalize(dir).x);
        toSort[i].originalIndex = (uint32_t)i;
    }
    auto blockCount = (toSort.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    naiveXYCluster(ArrayView<DirectionalSampleToSort>(toSort), blockCount);
    for (size_t i = 0; i < toSort.size(); ++i) {
        remap[toSort[i].originalIndex] = (uint32_t)i;
    }
}

void getEccentricityRemap(std::vector<float>& eccentricityRemap,
                          const std::vector<float>& ringEccentricities,
                          float maxEccentricityRadians,
                          size_t mapSize) {
    eccentricityRemap.resize(mapSize);
    for (int i = 0; i < eccentricityRemap.size(); ++i) {
        float eccentricity = ((i + 0.5f) / eccentricityRemap.size()) * maxEccentricityRadians;
        int j = 0;
        bool inBetween = false;
        while (j < ringEccentricities.size()) {
            if (eccentricity < ringEccentricities[j]) {
                inBetween = true;
                break;
            }
            ++j;
        }
        // TODO: fit spline instead of piecewise linear approx
        if (j == 0) {
            eccentricityRemap[i] = ((eccentricity / ringEccentricities[0]) * 0.5f) / ringEccentricities.size();
        } else if (inBetween) {
            float lower = ringEccentricities[j - 1];
            float higher = ringEccentricities[j];
            float alpha = (eccentricity - lower) / (higher - lower);
            eccentricityRemap[i] = (((j - 1) * (1.0f - alpha) + j * alpha) + 0.5f) / ringEccentricities.size();
        } else {
            size_t lastIndex = ringEccentricities.size() - 1;
            float lastDiff = (ringEccentricities[lastIndex] - ringEccentricities[lastIndex - 1]);
            float extrapolation = (eccentricity - ringEccentricities[lastIndex]) / lastDiff;
            eccentricityRemap[i] =
                min(1.0f, (ringEccentricities.size() - 0.5f + extrapolation) / ringEccentricities.size());
        }
    }
}

void generateEyeSpacePolarFoveatedSampleData(FoveatedSampleData& foveatedSampleData,
                                             std::vector<vector2ui>& polarRemapToPixel,
                                             std::vector<float>& ringEccentricities,
                                             std::vector<float>& eccentricityRemap,
                                             size_t& samplesPerRing,
                                             RayCasterSpecification::FoveatedSamplePattern pattern) {
    if (foveatedSampleData.eyeSpaceSamples.size() == 0) {
        DynamicArray<DirectionSample> unsortedEyeSpaceSamples = getEyeSpacePolarFoveatedSamples(
            ringEccentricities, samplesPerRing, pattern.degreeTrackingError, pattern.minMAR, pattern.maxMAR,
            pattern.maxFOVDegrees, pattern.marSlope, pattern.fovealMARDegrees, pattern.zenithJitterStrength,
            pattern.ringJitterStrength);
        printf("Generated eyes space foveated samples: %d, in %dx%d polar grid (azimuth x zenith)\n",
               uint32_t(unsortedEyeSpaceSamples.size()), uint32_t(samplesPerRing),
               uint32_t(unsortedEyeSpaceSamples.size() / samplesPerRing));
        float maxEccentricityRadians = pattern.maxFOVDegrees * RadiansPerDegree;
        getEccentricityRemap(eccentricityRemap, ringEccentricities, maxEccentricityRadians, 10000);
        {
            DynamicArray<uint32_t> oldToNewRemap(unsortedEyeSpaceSamples.size());
            generateRemapForFoveatedSamples(unsortedEyeSpaceSamples, oldToNewRemap);
            foveatedSampleData.eyeSpaceSamples = DynamicArray<DirectionSample>(unsortedEyeSpaceSamples.size());
            for (size_t i = 0; i < unsortedEyeSpaceSamples.size(); ++i) {
                foveatedSampleData.eyeSpaceSamples[oldToNewRemap[i]] = unsortedEyeSpaceSamples[i];
            }
            polarRemapToPixel.resize(unsortedEyeSpaceSamples.size());
            for (size_t i = 0; i < unsortedEyeSpaceSamples.size(); ++i) {
                polarRemapToPixel[oldToNewRemap[i]] = {(uint32_t)(i % samplesPerRing), (uint32_t)(i / samplesPerRing)};
            }
        }
        {
            foveatedSampleData.precomputedEyeSpaceSamples =
                DynamicArray<PrecomputedDirectionSample>(foveatedSampleData.eyeSpaceSamples.size());
            for (size_t i = 0; i < unsortedEyeSpaceSamples.size(); ++i) {
                auto& p = foveatedSampleData.precomputedEyeSpaceSamples[i];
                p.center = foveatedSampleData.eyeSpaceSamples[i].direction;
                p.d1 = p.center + foveatedSampleData.eyeSpaceSamples[i].azimuthalDifferential;
                p.d2 = p.center + foveatedSampleData.eyeSpaceSamples[i].zenithDifferential;
            }
        }
        auto blockCount = (unsortedEyeSpaceSamples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // Allocate the most you will ever need to prevent per-frame allocation
        foveatedSampleData.samples.blockFrusta2D = DynamicArray<RayPacketFrustum2D>(blockCount);
        foveatedSampleData.samples.tileFrusta2D = DynamicArray<RayPacketFrustum2D>(blockCount * TILES_PER_BLOCK);
        foveatedSampleData.samples.blockFrusta3D = DynamicArray<RayPacketFrustum3D>(blockCount);
        foveatedSampleData.samples.tileFrusta3D = DynamicArray<RayPacketFrustum3D>(blockCount * TILES_PER_BLOCK);
    }
}

void polarSpaceFoveatedSetup(Raycaster* raycaster) {

    for (auto& camera : raycaster->_cameras) {
        if (!camera->getEnabled())
            continue;
        // Generate eye space samples if necessary
        if (camera->_foveatedSampleData.eyeSpaceSamples.size() == 0) {
            size_t samplesPerRing;

            std::vector<float> eccentricityRemap;
            std::vector<float> ringEccentricities;
            generateEyeSpacePolarFoveatedSampleData(camera->_foveatedSampleData, camera->_polarRemapToPixel,
                                                    ringEccentricities, eccentricityRemap, samplesPerRing,
                                                    raycaster->_spec.foveatedSamplePattern);
            float maxEccentricityRadians = raycaster->_spec.foveatedSamplePattern.maxFOVDegrees * RadiansPerDegree;
            size_t paddedSampleCount =
                ((camera->_foveatedSampleData.eyeSpaceSamples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            RegisterPolarFoveatedSamples(camera.get(), camera->_polarRemapToPixel, maxEccentricityRadians,
                                         ringEccentricities, eccentricityRemap, uint32_t(samplesPerRing),
                                         uint32_t(paddedSampleCount));
            UpdateEyeSpaceFoveatedSamples(camera.get(), camera->_foveatedSampleData.precomputedEyeSpaceSamples);
            camera->_foveatedSampleData.simpleBlockFrusta =
                DynamicArray<SimpleRayFrustum>(camera->_foveatedSampleData.samples.blockFrusta3D.size());
            camera->_foveatedSampleData.simpleTileFrusta =
                DynamicArray<SimpleRayFrustum>(camera->_foveatedSampleData.samples.tileFrusta3D.size());
        }
    }
}

} // namespace hvvr
