#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dynamic_array.h"
#include "graphics_types.h"
#include <cassert>

namespace hvvr {
struct EccentricityMap;
struct Sample {
    vector2ui pixelLocation;
    vector2 position;

    // Encoding of 2D ellipse, designed for fast computation
    // majorAxisLength is used directly for conservative bounds
    // they can be computed without costly atan2 operations (like polar coordinates)
    // and its fairly cheap to compute the major axis (just a reciprocal sqrt and a few muls)
    struct Extents {
        vector2 minorAxis;
        float majorAxisLength;
    } extents;
};

struct UnpackedSample {
    vector2 center;
    vector2 majorAxis;
    vector2 minorAxis;
};

UnpackedSample unpackSample(Sample s);

// Header for binary file of Samples
struct SampleFileHeader {
    uint32_t magic = 0x0CD1; // Oculus Distortion
    uint32_t version = 0;
    uint32_t reserved0;
    uint32_t sampleCount;
};

// Save to file using a simple binary format
void loadSamples(hvvr::DynamicArray<hvvr::Sample>& samples, const std::string& filename);

// Load from file using a simple binary format
void saveSamples(const std::vector<hvvr::Sample>& samples, const std::string& filename);


struct SortedSample : Sample {
    uint32_t channel;

    SortedSample(){};
    SortedSample(const Sample& sample, uint32_t channel) : Sample(sample), channel(channel){};
};

DynamicArray<Sample> getGridSamples(size_t width, size_t height);

struct DirectionalBeam {
    vector3 centerRay;
    vector3 du;
    vector3 dv;
};

/*

    Generate eye-space samples matching the eyes resolution with density using the linear model from
    https://www.microsoft.com/en-us/research/wp-content/uploads/2012/11/foveated_final15.pdf

    We generate the actual samples in concentric circles, with the spacing between the circles equal to the calculated
   minimum angular resolution (MAR) of the inner circle, and samples are spaced on each circle so that they fulfill the
   MAR

    The linear model is:
    w = m e + w_0
    e is the eccentricity angle
    m is the linear slope which measures acuity falloff
    w_0 is the MAR of the fovea
    w is the calculated MAR of the circle with eccentricity e

    The Microsoft paper calculates two values of m:
    m_A = 0.0275
    m_B = 0.0220
    where m_B is the more conservative value.
    They use
    w_0 = 1/48 degrees (20/10 vision is 1/60 degrees)

    \param maxEyeTrackingUncertaintyDegrees If set to > 0 then e acts like max(e - maxEyeTrackingUncertaintyDegrees, 0).
   This ensures that we conservatively calculate the sampling positions so that we don't accidentally undersample
   regions of the screen due to error in eye tracking. The higher this value, the more samples are generated.
    \param maxFOVDegrees is the diagonal FOV of the display, we generate samples such that even if you are looking in
   the far corner of the display, there is one ring of samples beyond the opposite corner of the display, so that the
   entire screen can still be reconstructed from the foveate samples without issue.
    \param marSlope corresponds to m, we default it to the more conservative m_B
    \param fovealMARDegrees corresponds to w_0, we default it to 1/60 degrees (20/10) to be even more conservative than
   the microsoft paper

   Consider replacing this space with a transformed log-polar coordinate system:
   Weiman, Chaikin, Logarithmic Spiral Grids for Image Processing and Display, Computer Graphics and Image Processing
   11, 197–226 (1979).
*/
DynamicArray<DirectionalBeam> getEyeSpacePolarFoveatedSamples(size_t& samplesPerRing,
                                                              EccentricityMap& emap,
                                                              float maxEyeTrackingUncertaintyDegrees,
                                                              float maxFOVDegrees,
                                                              float marSlope,
                                                              float fovealMARDegrees);

} // namespace hvvr
