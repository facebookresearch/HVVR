/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "foveated.h"
#include "camera.h"
#include "constants_math.h"
#include "gpu_camera.h"
#include "raycaster.h"
#include "sample_clustering.h"


namespace hvvr {

void generateRemapForFoveatedSamples(ArrayView<DirectionalBeam> unsortedSamples,
                                     ArrayView<uint32_t> remap) { // Create a remapping from the original directional
                                                                  // samples to ones that are binned decently

    struct DirectionalSampleToSort {
        vector2 position; // eccentricity,theta (octahedral is a bad match for the naive clustering algorithm)
        uint32_t originalIndex;
    };
    // There is a massive performance difference between using eccentricity for the y coordinate and the x coordinate in
    // this clustering This points to this clustering being unstable and we might want a better algorithm. ~3.5x
    // performance improvement in intersection when eccentricity is the y coordinate, which leads to better distribution
    // (for default foveation settings at least)
    DynamicArray<DirectionalSampleToSort> toSort(unsortedSamples.size());
    for (size_t i = 0; i < toSort.size(); ++i) {
        vector3 v = unsortedSamples[i].centerRay;
        // Eccentricity angle
        toSort[i].position.y = acosf(-v.z);
        vector3 dir = normalize(vector3(v.x, v.y, 0.0f));
        // Angle of rotation about z, measured from x
        toSort[i].position.x = atan2f(dir.y, dir.x);
        toSort[i].originalIndex = (uint32_t)i;
    }
    auto blockCount = (toSort.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    naiveXYCluster(ArrayView<DirectionalSampleToSort>(toSort), blockCount);
    for (size_t i = 0; i < toSort.size(); ++i) {
        remap[toSort[i].originalIndex] = (uint32_t)i;
    }
}

void generateEyeSpacePolarFoveatedSampleData(FoveatedSampleData& foveatedSampleData,
                                             std::vector<vector2ui>& polarRemapToPixel,
                                             EccentricityMap& eccentricityMap,
                                             size_t& samplesPerRing,
                                             RayCasterSpecification::FoveatedSamplePattern pattern) {
    if (foveatedSampleData.samples.directionalBeams.size() == 0) {
        DynamicArray<DirectionalBeam> unsortedEyeSpaceSamples =
            getEyeSpacePolarFoveatedSamples(samplesPerRing, eccentricityMap, pattern.degreeTrackingError,
                                            pattern.maxFOVDegrees, pattern.marSlope, pattern.fovealMARDegrees);

        printf("Generated eyes space foveated samples: %d, in %dx%d polar grid (azimuth x zenith)\n",
               uint32_t(unsortedEyeSpaceSamples.size()), uint32_t(samplesPerRing),
               uint32_t(unsortedEyeSpaceSamples.size() / samplesPerRing));
        {
            DynamicArray<uint32_t> oldToNewRemap(unsortedEyeSpaceSamples.size());
            generateRemapForFoveatedSamples(unsortedEyeSpaceSamples, oldToNewRemap);
            foveatedSampleData.samples.directionalBeams = DynamicArray<DirectionalBeam>(unsortedEyeSpaceSamples.size());
            polarRemapToPixel.resize(unsortedEyeSpaceSamples.size());
            for (size_t i = 0; i < unsortedEyeSpaceSamples.size(); ++i) {
                foveatedSampleData.samples.directionalBeams[oldToNewRemap[i]] = unsortedEyeSpaceSamples[i];
                polarRemapToPixel[oldToNewRemap[i]] = {(uint32_t)(i % samplesPerRing), (uint32_t)(i / samplesPerRing)};
            }
        }
        auto blockCount = (unsortedEyeSpaceSamples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // Allocate the most you will ever need to prevent per-frame allocation
        foveatedSampleData.samples.blockFrusta3D = DynamicArray<Frustum>(blockCount);
        foveatedSampleData.samples.tileFrusta3D = DynamicArray<Frustum>(blockCount * TILES_PER_BLOCK);
        foveatedSampleData.blockCount = blockCount;
    }
}

void polarSpaceFoveatedSetup(Raycaster* raycaster) {
    for (auto& camera : raycaster->_cameras) {
        if (!camera->getEnabled())
            continue;
        // Generate eye space samples if necessary
        BeamBatch& beamHierarchy = camera->_foveatedSampleData.samples;
        DynamicArray<DirectionalBeam>& cameraBeams = beamHierarchy.directionalBeams;
        if (cameraBeams.size() == 0) {
            size_t samplesPerRing;
            EccentricityMap eccentricityMap;
            generateEyeSpacePolarFoveatedSampleData(camera->_foveatedSampleData, camera->_polarRemapToPixel,
                                                    eccentricityMap, samplesPerRing,
                                                    raycaster->_spec.foveatedSamplePattern);
            float maxEccentricityRadians = raycaster->_spec.foveatedSamplePattern.maxFOVDegrees * RadiansPerDegree;
            size_t paddedSampleCount = ((cameraBeams.size() + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            camera->_gpuCamera->registerPolarFoveatedSamples(camera->_polarRemapToPixel, maxEccentricityRadians,
                                                             eccentricityMap, uint32_t(samplesPerRing),
                                                             uint32_t(paddedSampleCount));
            camera->_gpuCamera->updateEyeSpaceFoveatedSamples(beamHierarchy);

            size_t blockCount = beamHierarchy.blockFrusta3D.size();
            size_t tileCount = beamHierarchy.tileFrusta3D.size();
            if (blockCount != camera->_cpuHierarchy._blockFrusta.size()) {
                camera->_cpuHierarchy._blockFrusta = DynamicArray<Frustum>(blockCount);
            }
            if (tileCount != camera->_cpuHierarchy._tileFrusta.size()) {
                camera->_cpuHierarchy._tileFrusta = DynamicArray<Frustum>(tileCount);
            }
        }
    }
}

} // namespace hvvr
