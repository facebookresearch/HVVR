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
#include "foveated.h"
#include "frusta.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "gpu_foveated.h"
#include "gpu_samples.h"
#include "graphics_types.h"
#include "kernel_constants.h"
#include "memory_helpers.h"
#include "shading_helpers.h"
#include "vector_math.h"


namespace hvvr {

// TODO: move to helper header, make generic;
// potentially optimize down to 4 taps using smart tap placement
CUDA_DEVICE vector4 bicubic(Texture2D tex, vector2 coord) {
    vector2 pixCoord = coord * vector2(tex.width, tex.height);
    vector2 pixCenter = vector2(floorf(pixCoord.x - 0.5f), floorf(pixCoord.y - 0.5f)) + 0.5f;

    vector2 iDim = vector2(1.0f / tex.width, 1.0f / tex.height);

    // fractionalOffset
    vector2 f = pixCoord - pixCenter;
    vector2 f2 = f * f;
    vector2 f3 = f2 * f;
    vector2 omf2 = (vector2(1.0f) - f) * (vector2(1.0f) - f);
    vector2 omf3 = omf2 * (vector2(1.0f) - f);
    float sixth = (1.0f / 6.0f);
    // B-spline
    vector2 w[4] = {sixth * omf3, sixth * (4.0f + 3.0f * f3 - 6.0f * f2), sixth * (4.0f + 3.0f * omf3 - 6.0f * omf2),
                    sixth * f3};

    vector2 tc[4] = {pixCenter + vector2(-1), pixCenter, pixCenter + vector2(1), pixCenter + vector2(2)};

    vector4 result = vector4(0.0f);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            result += vector4(tex2D<float4>(tex.d_texObject, tc[x].x * iDim.x, tc[y].y * iDim.y)) * w[x].x * w[y].y;
        }
    }
    return result;
}


CUDA_DEVICE vector2 directionToSampleSpaceSample(const matrix3x3& eyeSpaceToSampleSpaceMatrix,
                                                 const vector3& direction) {
    vector3 sampleSpaceSample = eyeSpaceToSampleSpaceMatrix * direction;
    float invZ = (1.0f / sampleSpaceSample.z);
    return vector2(sampleSpaceSample.x * invZ, sampleSpaceSample.y * invZ);
}

// Inclusive contains
CUDA_DEVICE_INL bool rectContains(const FloatRect r, const vector2 p) {
    return (p.x >= r.lower.x) && (p.x <= r.upper.x) && (p.y >= r.lower.y) && (p.y <= r.upper.y);
}

CUDA_KERNEL void TransformFoveatedSamplesToSampleSpaceKernel(
    const matrix3x3 eyeSpaceToSampleSpaceMatrix,
    const matrix3x3 eyeSpaceToCameraSpace,
    const FloatRect cullRect,
    const PrecomputedDirectionSample* precomputedEyeSpaceSamples,
    SampleInfo sampleInfo,
    int* remap,
    const uint32_t sampleCount) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sampleCount) {
        vector3 direction = precomputedEyeSpaceSamples[index].center;
        vector2 c = {CUDA_INF, CUDA_INF};
        if ((eyeSpaceToCameraSpace * direction).z < 0.0f) {
            vector2 s = directionToSampleSpaceSample(eyeSpaceToSampleSpaceMatrix, direction);
            if (rectContains(cullRect, s)) {
                c = s;
                vector2 d1 =
                    directionToSampleSpaceSample(eyeSpaceToSampleSpaceMatrix, precomputedEyeSpaceSamples[index].d1) - c;
                vector2 d2 =
                    directionToSampleSpaceSample(eyeSpaceToSampleSpaceMatrix, precomputedEyeSpaceSamples[index].d2) - c;
                float sqExtent1 = dot(d1, d1);
                float sqExtent2 = dot(d2, d2);
                ;
                Sample::Extents extent;
                if (sqExtent1 > sqExtent2) {
                    extent.minorAxis = d2;
                    extent.majorAxisLength = sqrtf(sqExtent1);
                } else if (sqExtent2 == 0) {
                    extent.minorAxis.x = 0;
                    extent.minorAxis.y = 0;
                    extent.majorAxisLength = 0;
                } else { // sqExtent2 >= sqExtent1, sqExtent2 != 0
                    extent.minorAxis = d1;
                    extent.majorAxisLength = sqrtf(sqExtent2);
                }
                remap[index] = (int)index;
                sampleInfo.extents[index] = extent;
            }
        }
        sampleInfo.centers[index] = c;
    }
}

void TransformFoveatedSamplesToSampleSpace(const matrix3x3& eyeSpaceToSampleSpaceMatrix,
                                           const matrix3x3& eyeSpaceToCameraSpaceMatrix,
                                           const FloatRect& sampleBounds,
                                           const PrecomputedDirectionSample* d_precomputedEyeSpaceSamples,
                                           SampleInfo sampleInfo,
                                           int* d_remap,
                                           const uint32_t sampleCount,
                                           cudaStream_t stream) {
    KernelDim dim = KernelDim(sampleCount, CUDA_GROUP_SIZE);
    TransformFoveatedSamplesToSampleSpaceKernel<<<dim.grid, dim.block, 0, stream>>>(
        eyeSpaceToSampleSpaceMatrix, eyeSpaceToCameraSpaceMatrix, sampleBounds, d_precomputedEyeSpaceSamples,
        sampleInfo, d_remap, sampleCount);
}

struct EccentricityToTexCoordMapping {
    float maxEccentricityRadians;
    float* forwardMap;
    float forwardMapSize;
    float* backwardMap;
    float backwardMapSize;
};

void GPUCamera::getEccentricityMap(EccentricityToTexCoordMapping& map) const {
    map.maxEccentricityRadians = maxEccentricityRadians;
    map.forwardMap = d_eccentricityCoordinateMap;
    map.forwardMapSize = (float)d_eccentricityCoordinateMap.size();
    map.backwardMap = (float*)d_ringEccentricities;
    map.backwardMapSize = (float)d_ringEccentricities.size();
}

CUDA_DEVICE float bilinearRead1D(float coord, float* map, float mapSize) {
    float pixelCoord = coord * mapSize - 0.5f;
    float integralPart = floor(pixelCoord);
    int maxCoord = (int)mapSize - 1;
    int lowerCoord = clamp((int)integralPart, 0, maxCoord);
    int upperCoord = clamp((int)integralPart + 1, 0, maxCoord);
    float alpha = pixelCoord - integralPart;
    return lerp(map[lowerCoord], map[upperCoord], alpha);
}

// eccentricity is in the range [0,maxEccentricityRadians]
CUDA_DEVICE float eccentricityToTexCoord(float eccentricity, EccentricityToTexCoordMapping eToTexMap) {
    float normalizedE = eccentricity / eToTexMap.maxEccentricityRadians;
    return bilinearRead1D(normalizedE, eToTexMap.forwardMap, eToTexMap.forwardMapSize);
}

CUDA_DEVICE vector2 getNormalizedCoord(int x, int y, int width, int height) {
    return vector2(((float)x + 0.5f) / (float)width, ((float)y + 0.5f) / (float)height);
}

// TODO: Canonicalize the negations...
// Aligned along z axis
CUDA_DEVICE vector3 angularEyeCoordToDirection(float theta, float e) {
    float z = -cosf(e);
    float xyLength = sqrtf(1.0f - z * z);
    vector2 xy = vector2(cosf(-theta), sinf(-theta)) * xyLength;
    return {xy.x, xy.y, z};
}
CUDA_DEVICE void eyeSpaceDirectionToAngularEyeCoord(vector3 dir, float& theta, float& eccentricity) {
    // TODO: get rid of transcendentals in calculation
    eccentricity = acosf(-dir.z);
    // Angle of rotation about z, measured from x
    theta = -atan2f(dir.y, dir.x);
}

CUDA_DEVICE void polarTextureCoordToAngularEyeCoord(vector2 coord,
                                                    EccentricityToTexCoordMapping eToTexMap,
                                                    float& theta,
                                                    float& eccentricity) {
    eccentricity = bilinearRead1D(coord.y, eToTexMap.backwardMap, eToTexMap.backwardMapSize);
    theta = (2.0f * Pi * coord.x) - Pi;
}
CUDA_DEVICE vector2 angularEyeCoordToPolarTextureCoord(float theta,
                                                       float eccentricity,
                                                       EccentricityToTexCoordMapping eToTexMap) {
    float x = (theta + Pi) / (2.0f * Pi);
    float y = eccentricityToTexCoord(eccentricity, eToTexMap);
    return {x, y};
}

CUDA_DEVICE void computeMoments3x3Window(
    cudaTextureObject_t tex, vector2 coord, vector2 invDim, vector4& m_1, vector4& m_2) {
    float offsets[3] = {-1.0f, 0.0f, 1.0f};
    m_1 = vector4(0.0f);
    m_2 = vector4(0.0f);
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            vector4 c(tex2D<float4>(tex, coord.x + (offsets[x] * invDim.x), coord.y + (offsets[y] * invDim.y)));
            m_1 += c;
            m_2 += c * c;
        }
    }
    float weight = 1.0f / 9.0f;
    m_1 *= weight;
    m_2 *= weight;
}

CUDA_DEVICE vector4 sqrtf(vector4 v) {
    return vector4(v.x, v.y, v.z, v.w);
}

CUDA_DEVICE vector4 clampToNeighborhood(
    vector4 oldValue, cudaTextureObject_t tex, vector2 coord, vector2 invDim, TemporalFilterSettings settings) {
    vector4 m_1, m_2;
    computeMoments3x3Window(tex, coord, invDim, m_1, m_2);
    vector4 stdDev = sqrtf(m_2 - (m_1 * m_1));
    // Arbitrary
    float scaleFactor = settings.stddevMultiplier;
    vector4 minC = m_1 - (stdDev * scaleFactor);
    vector4 maxC = m_1 + (stdDev * scaleFactor);
    return clamp(oldValue, minC, maxC);
}

template <PixelFormat PixelFormat>
CUDA_KERNEL void FoveatedPolarToScreenSpaceKernel(Texture2D polarImage,
                                                  Texture2D resultTexture,
                                                  GPUImage resultImage,
                                                  matrix3x3 sampleSpaceToEyeSpaceMatrix,
                                                  EccentricityToTexCoordMapping eToTexMap,
                                                  Texture2D previousImage,
                                                  Texture2D polarFoveatedTMaxImage,
                                                  matrix4x4 eyeSpaceToPreviousSampleSpaceMatrix,
                                                  TemporalFilterSettings settings) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < resultImage.width() && j < resultImage.height()) {
        vector2 normalizedCoord = getNormalizedCoord(i, j, resultImage.width(), resultImage.height());
        vector3 sampleSpacePoint = vector3(normalizedCoord, 1.0f);
        vector3 eyeSpaceDirection = normalize(sampleSpaceToEyeSpaceMatrix * sampleSpacePoint);

        float theta, eccentricity;
        eyeSpaceDirectionToAngularEyeCoord(eyeSpaceDirection, theta, eccentricity);

        vector2 coord = angularEyeCoordToPolarTextureCoord(theta, eccentricity, eToTexMap);

        vector4 newValue = bicubic(polarImage, coord);
        vector4 result = newValue;
        vector4 surfaceResult = result;
        if (!settings.inPolarSpace) {
            float tValue = tex2D<float>(polarFoveatedTMaxImage.d_texObject, coord.x, coord.y);
            // Clamp infinity. TODO: something more robust
            tValue = min(tValue, 1000.0f);

            vector3 currentEyePosition = angularEyeCoordToDirection(theta, eccentricity) * tValue;

            vector4 prevSamplePosition = eyeSpaceToPreviousSampleSpaceMatrix * vector4(currentEyePosition, 1.0f);
            vector2 oldTexCoord = vector2(prevSamplePosition.x, prevSamplePosition.y) * (1.0f / prevSamplePosition.z);

            float alpha = settings.alpha;
            vector4 oldValue = newValue;
            if (oldTexCoord.x > 0 && oldTexCoord.y > 0 && oldTexCoord.x < 1 && oldTexCoord.y < 1 && tValue > 0) {
                oldValue = vector4(tex2D<float4>(previousImage.d_texObject, oldTexCoord.x, oldTexCoord.y));
            }
            // TODO: could compute the moments in a prepass
            vector2 invDim = vector2(1.0f / polarImage.width, 1.0f / polarImage.height);
            vector4 clampedOldValue = clampToNeighborhood(oldValue, polarImage.d_texObject, coord, invDim, settings);
            surfaceResult = alpha * newValue + (1.0f - alpha) * clampedOldValue;
            result = surfaceResult;
        }

        // DEBUG CODE:
        // surfaceResult = vector4(tex2D<float4>(polarImage.d_texObject, normalizedCoord.x, normalizedCoord.y));
        // result = surfaceResult;

        if (PixelFormat == PixelFormat::RGBA32F) {
            vector4* output = (vector4*)resultImage.data();
            output[resultImage.stride() * j + i] = result;
            if (!settings.inPolarSpace) {
                surf2Dwrite(float4(surfaceResult), resultTexture.d_surfaceObject, i * sizeof(float4), j);
            }
        } else {
            uint32_t* output = (uint32_t*)resultImage.data();
            output[resultImage.stride() * j + i] = ToColor4Unorm8SRgb(result);
            if (!settings.inPolarSpace) {
                surf2Dwrite(ToColor4Unorm8SRgb(surfaceResult), resultTexture.d_surfaceObject, i * sizeof(uchar4), j);
            }
        }
    }
}

template <PixelFormat PixelFormat>
CUDA_KERNEL void FoveatedTemporalFilterKernel(Texture2D rawImage,
                                              Texture2D previousImage,
                                              Texture2D polarFoveatedTMaxImage,
                                              EccentricityToTexCoordMapping eToTexMap,
                                              matrix4x4 eyeToEyePrevious,
                                              TemporalFilterSettings settings,
                                              Texture2D resultImage) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rawImage.width && j < rawImage.height) {
        vector2 invDim = vector2(1.0f / rawImage.width, 1.0f / rawImage.height);
        vector2 coord = vector2(i + 0.5f, j + 0.5f) * invDim;
        float tValue = tex2D<float>(polarFoveatedTMaxImage.d_texObject, coord.x, coord.y);
        vector4 newValue(tex2D<float4>(rawImage.d_texObject, coord.x, coord.y));
        vector4 result = newValue;
        if (tValue < CUDA_INF) {
            float theta, e;
            polarTextureCoordToAngularEyeCoord(coord, eToTexMap, theta, e);

            vector3 currentEyePosition = angularEyeCoordToDirection(theta, e) * tValue;

            vector4 prevEyePosition = eyeToEyePrevious * vector4(currentEyePosition, 1.0f);

            float prevTheta, prevE;
            eyeSpaceDirectionToAngularEyeCoord(normalize(vector3(prevEyePosition)), prevTheta, prevE);

            vector2 oldTexCoord = angularEyeCoordToPolarTextureCoord(prevTheta, prevE, eToTexMap);

            float alpha = settings.alpha;

            vector4 oldValue(tex2D<float4>(previousImage.d_texObject, oldTexCoord.x, oldTexCoord.y));

            // TODO: could compute the moments in a prepass
            vector4 clampedOldValue = clampToNeighborhood(oldValue, rawImage.d_texObject, coord, invDim, settings);
            result = alpha * newValue + (1.0f - alpha) * clampedOldValue;
        }

        // vector2 diff = (oldTexCoord - coord);
        // result = newValue;// {tValue, -tValue, 0.0f, 0.0f};
        if (PixelFormat == PixelFormat::RGBA32F) {
            surf2Dwrite(float4(result), resultImage.d_surfaceObject, i * sizeof(float4), j);
        } else {
            surf2Dwrite(ToColor4Unorm8SRgb(result), resultImage.d_surfaceObject, i * sizeof(uchar4), j);
        }
    }
}

CUDA_DEVICE float getEccentricity(unsigned i, unsigned j, Texture2D tex, matrix3x3 sampleSpaceToEyeSpaceMatrix) {
    vector2 normalizedCoord = getNormalizedCoord(i, j, tex.width, tex.height);
    vector3 eyeSpaceDirection =
        normalize(sampleSpaceToEyeSpaceMatrix * vector3(normalizedCoord.x, normalizedCoord.y, 1.0f));
    float theta, eccentricity;
    eyeSpaceDirectionToAngularEyeCoord(eyeSpaceDirection, theta, eccentricity);
    return eccentricity;
}

CUDA_DEVICE vector4 texelFetch(Texture2D tex, unsigned i, unsigned j) {
    vector2 coord = getNormalizedCoord(i, j, tex.width, tex.height);
    return vector4(tex2D<float4>(tex.d_texObject, coord.x, coord.y));
}

template <PixelFormat PixelFormat>
CUDA_KERNEL void SeparableFilterUsingEccentricity(Texture2D output,
                                                  Texture2D input,
                                                  vector2i step,
                                                  ContrastEnhancementSettings settings,
                                                  matrix3x3 sampleSpaceToEyeSpaceMatrix) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < output.width && j < output.height) {
        float eccentricity = getEccentricity(i, j, output, sampleSpaceToEyeSpaceMatrix);
        // TODO: compute filter radius
        int filterRadius = 5;

        vector4 valueSum = vector4(0.0f, 0.0f, 0.0f, 0.0f);
        float weightSum = 0.0f;
        for (int R = -filterRadius; R <= filterRadius; ++R) {
            vector2i tapLoc(clamp((int)i + R * step.x, (int)0, (int)output.width - 1),
                            clamp((int)j + R * step.y, (int)0, (int)output.height - 1));
            // TODO: compute filter weight
            float weight = 1.0f;
            valueSum += texelFetch(input, tapLoc.x, tapLoc.y) * weight;
            weightSum += weight;
        }
        vector4 result = valueSum / weightSum;
        surf2Dwrite(ToColor4Unorm8SRgb(result), output.d_surfaceObject, i * sizeof(uchar4), j);
    }
}

/** From https://research.nvidia.com/sites/default/files/publications/supplementary.pdf
    They had a vec2 for sigma, we currently have a float so don't need to take its length */
__device__ vector4 enhanceContrast(vector4 pix, vector4 pmean, float sigma, float f_e) {
    // computer amount of contrast enhancement
    // based on degree of foveation (sigma)
    float cScale = 1.f + sigma * f_e;
    vector4 scaledColor = pmean + (pix - pmean) * cScale;
    return clamp(scaledColor, 0.0f, 1.0f);
}

template <PixelFormat PixelFormat>
CUDA_KERNEL void FinishConstrastEnhancement(GPUImage resultImage,
                                            Texture2D unfilteredTexture,
                                            Texture2D filteredTexture,
                                            ContrastEnhancementSettings settings,
                                            matrix3x3 sampleSpaceToEyeSpaceMatrix) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < resultImage.width() && j < resultImage.height()) {
        // TODO: compute sigma
        float eccentricity = getEccentricity(i, j, unfilteredTexture, sampleSpaceToEyeSpaceMatrix);
        float sigma = 1.0f;

        vector4 pix = texelFetch(unfilteredTexture, i, j);
        vector4 pmean = texelFetch(filteredTexture, i, j);

        vector4 result = enhanceContrast(pix, pmean, sigma, settings.f_e);

        uint32_t* output = (uint32_t*)resultImage.data();
        output[resultImage.stride() * j + i] = ToColor4Unorm8SRgb(result);
    }
}

static void textureCopy(Texture2D dst, Texture2D src) {
    assert(dst.width == src.width && dst.height == src.height && dst.format == src.format);
    cutilSafeCall(cudaMemcpy2DArrayToArray(dst.d_rawMemory, 0, 0, src.d_rawMemory, 0, 0, dst.width * dst.elementSize,
                                           dst.height));
}

void GPUCamera::foveatedPolarTemporalFilter(const matrix4x4& eyeToEyePrevious) {
    size_t width = polarFoveatedImage.width;
    size_t height = polarFoveatedImage.height;
    KernelDim dim = KernelDim(width, height, CUDA_GROUP_WIDTH, CUDA_GROUP_HEIGHT);

    EccentricityToTexCoordMapping eToTexMap;
    getEccentricityMap(eToTexMap);

    switch (outputModeToPixelFormat(outputMode)) {
        case PixelFormat::RGBA32F:
            FoveatedTemporalFilterKernel<PixelFormat::RGBA32F><<<dim.grid, dim.block, 0, stream>>>(
                rawPolarFoveatedImage, previousPolarFoveatedImage, polarFoveatedDepthImage, eToTexMap, eyeToEyePrevious,
                temporalFilterSettings, polarFoveatedImage);
            break;
        case PixelFormat::RGBA8_SRGB:
            FoveatedTemporalFilterKernel<PixelFormat::RGBA8_SRGB><<<dim.grid, dim.block, 0, stream>>>(
                rawPolarFoveatedImage, previousPolarFoveatedImage, polarFoveatedDepthImage, eToTexMap, eyeToEyePrevious,
                temporalFilterSettings, polarFoveatedImage);
            break;
        default:
            assert(false);
    }
    // TODO: could ping-pong buffers to save a copy
    textureCopy(previousPolarFoveatedImage, polarFoveatedImage);
}

void GPUCamera::foveatedPolarToScreenSpace(const matrix4x4& eyeToEyePrevious,
                                           const matrix3x3& eyePreviousToSamplePrevious,
                                           const matrix3x3& sampleToEye) {
    bool filterInPolarSpace = temporalFilterSettings.inPolarSpace;
    if (filterInPolarSpace) {
        foveatedPolarTemporalFilter(eyeToEyePrevious);
    }

    KernelDim dim = KernelDim(resultImage.width(), resultImage.height(), CUDA_GROUP_WIDTH, CUDA_GROUP_HEIGHT);
    Texture2D currentPolarFoveatedImage = filterInPolarSpace ? polarFoveatedImage : rawPolarFoveatedImage;
    matrix4x4 eyeToSamplePrevious = matrix4x4(eyePreviousToSamplePrevious) * eyeToEyePrevious;

    EccentricityToTexCoordMapping eToTexMap;
    getEccentricityMap(eToTexMap);

    switch (outputModeToPixelFormat(outputMode)) {
        case PixelFormat::RGBA32F:
            FoveatedPolarToScreenSpaceKernel<PixelFormat::RGBA32F><<<dim.grid, dim.block, 0, stream>>>(
                currentPolarFoveatedImage, resultTexture, resultImage, sampleToEye, eToTexMap, previousResultTexture,
                polarFoveatedDepthImage, eyeToSamplePrevious, temporalFilterSettings);
            break;
        case PixelFormat::RGBA8_SRGB:
            FoveatedPolarToScreenSpaceKernel<PixelFormat::RGBA8_SRGB><<<dim.grid, dim.block, 0, stream>>>(
                currentPolarFoveatedImage, resultTexture, resultImage, sampleToEye, eToTexMap, previousResultTexture,
                polarFoveatedDepthImage, eyeToSamplePrevious, temporalFilterSettings);
            break;
        default:
            assert(false);
    }

    // TODO: could ping-pong buffers to save a copy
    textureCopy(previousResultTexture, resultTexture);

    if (contrastEnhancementSettings.enable) {
        assert(outputModeToPixelFormat(outputMode) == PixelFormat::RGBA8_SRGB);
        assert(filterInPolarSpace == false);
        // TODO: Probably cheaper to compute eccentricity once per pixel and store in
        // single channel 16F texture; and reuse in all three kernels
        SeparableFilterUsingEccentricity<PixelFormat::RGBA8_SRGB>
            <<<dim.grid, dim.block, 0, stream>>>(contrastEnhancementBuffers.horizontallyFiltered, resultTexture, {0, 1},
                                                 contrastEnhancementSettings, sampleToEye);
        SeparableFilterUsingEccentricity<PixelFormat::RGBA8_SRGB><<<dim.grid, dim.block, 0, stream>>>(
            contrastEnhancementBuffers.fullyFiltered, contrastEnhancementBuffers.horizontallyFiltered, {1, 0},
            contrastEnhancementSettings, sampleToEye);
        FinishConstrastEnhancement<PixelFormat::RGBA8_SRGB>
            <<<dim.grid, dim.block, 0, stream>>>(resultImage, resultTexture, contrastEnhancementBuffers.fullyFiltered,
                                                 contrastEnhancementSettings, sampleToEye);
    }
}

void GPUCamera::updateEyeSpaceFoveatedSamples(
    const ArrayView<PrecomputedDirectionSample> precomputedDirectionalSamples) {
    d_foveatedDirectionalSamples = GPUBuffer<PrecomputedDirectionSample>(precomputedDirectionalSamples.cbegin(),
                                                                         precomputedDirectionalSamples.cend());

    // Allocate and calculate eye-space frusta
    uint32_t blockCount = ((uint32_t)precomputedDirectionalSamples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_foveatedEyeSpaceTileFrusta = GPUBuffer<SimpleRayFrustum>(blockCount * TILES_PER_BLOCK);
    d_foveatedEyeSpaceBlockFrusta = GPUBuffer<SimpleRayFrustum>(blockCount);
    d_foveatedWorldSpaceTileFrusta = GPUBuffer<SimpleRayFrustum>(blockCount * TILES_PER_BLOCK);
    d_foveatedWorldSpaceBlockFrusta = GPUBuffer<SimpleRayFrustum>(blockCount);

    ComputeEyeSpaceFrusta(d_foveatedDirectionalSamples, d_foveatedEyeSpaceTileFrusta, d_foveatedEyeSpaceBlockFrusta);

    d_tileFrusta = GPUBuffer<GPURayPacketFrustum>(blockCount * TILES_PER_BLOCK);
    d_cullBlockFrusta = GPUBuffer<GPURayPacketFrustum>(blockCount);

    safeCudaEventDestroy(transferTileToCPUEvent);
    cutilSafeCall(cudaEventCreateWithFlags(&transferTileToCPUEvent, cudaEventDisableTiming));

    safeCudaFreeHost(tileFrustaPinned);
    safeCudaFreeHost(cullBlockFrustaPinned);

    cutilSafeCall(
        cudaMallocHost((void**)&tileFrustaPinned, sizeof(GPURayPacketFrustum) * blockCount * TILES_PER_BLOCK));
    cutilSafeCall(cudaMallocHost((void**)&cullBlockFrustaPinned, sizeof(GPURayPacketFrustum) * blockCount));

    safeCudaFreeHost(foveatedWorldSpaceTileFrustaPinned);
    safeCudaFreeHost(foveatedWorldSpaceBlockFrustaPinned);

    cutilSafeCall(cudaMallocHost((void**)&foveatedWorldSpaceTileFrustaPinned,
                                 sizeof(SimpleRayFrustum) * blockCount * TILES_PER_BLOCK));
    cutilSafeCall(cudaMallocHost((void**)&foveatedWorldSpaceBlockFrustaPinned, sizeof(SimpleRayFrustum) * blockCount));
}

void GPUCamera::updatePerFrameFoveatedData(const FloatRect& sampleBounds,
                                           const matrix3x3& cameraToSample,
                                           const matrix3x3& eyeToCamera,
                                           const matrix4x4& eyeToWorld) {
    validSampleCount = uint32_t(d_foveatedDirectionalSamples.size());
    SampleInfo sampleInfo(*this);
    uint32_t tileCount = uint32_t(d_tileFrusta.size());
    uint32_t blockCount = uint32_t(d_cullBlockFrusta.size());
    assert(d_foveatedWorldSpaceBlockFrusta.size() == d_cullBlockFrusta.size());

    ResetCullFrusta(d_cullBlockFrusta, d_tileFrusta, tileCount, blockCount, stream);

    matrix3x3 eyeToSample = cameraToSample * eyeToCamera;
    TransformFoveatedSamplesToSampleSpace(eyeToSample, eyeToCamera, sampleBounds, d_foveatedDirectionalSamples,
                                          sampleInfo, d_sampleRemap, validSampleCount, stream);

    CalculateSampleCullFrusta(d_cullBlockFrusta, d_tileFrusta, sampleInfo, validSampleCount, tileCount, blockCount,
                              stream);
    // Queue the copy back
    d_cullBlockFrusta.readbackAsync(cullBlockFrustaPinned, stream);
    d_tileFrusta.readbackAsync(tileFrustaPinned, stream);

    CalculateWorldSpaceFrusta(d_foveatedWorldSpaceBlockFrusta, d_foveatedWorldSpaceTileFrusta,
                              d_foveatedEyeSpaceBlockFrusta, d_foveatedEyeSpaceTileFrusta, eyeToWorld, blockCount,
                              tileCount, stream);
    // Queue the copy back
    d_foveatedWorldSpaceBlockFrusta.readbackAsync(foveatedWorldSpaceBlockFrustaPinned, stream);
    d_foveatedWorldSpaceTileFrusta.readbackAsync(foveatedWorldSpaceTileFrustaPinned, stream);

    cutilSafeCall(cudaEventRecord(transferTileToCPUEvent, stream));
}

} // namespace hvvr
