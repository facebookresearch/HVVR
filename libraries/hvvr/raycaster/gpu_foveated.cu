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
#include "gpu_samples.h"
#include "graphics_types.h"
#include "kernel_constants.h"
#include "memory_helpers.h"
#include "shading_helpers.h"
#include "vector_math.h"


namespace hvvr {

template <PixelFormat PixelFormat>
CUDA_DEVICE void writeSurface(vector4 val, Texture2D tex, unsigned int x, unsigned int y) {
    if (PixelFormat == PixelFormat::RGBA32F) {
        surf2Dwrite(float4(val), tex.d_surfaceObject, x * sizeof(float4), y);
    } else if (PixelFormat == PixelFormat::RGBA16) {
        surf2Dwrite(ToColor4Unorm16(val), tex.d_surfaceObject, x * sizeof(uint64_t), y);
    } else {
        surf2Dwrite(ToColor4Unorm8SRgb(val), tex.d_surfaceObject, x * sizeof(uchar4), y);
    }
}

// 4-tap B-spline, based on http://vec3.ca/bicubic-filtering-in-fewer-taps/
CUDA_DEVICE vector4 bicubicFast(Texture2D tex, vector2 coord) {
    vector2 pixCoord = coord * vector2(tex.width, tex.height);
    vector2 pixCenter = vector2(floorf(pixCoord.x - 0.5f), floorf(pixCoord.y - 0.5f)) + 0.5f;

    vector2 iDim = vector2(1.0f / tex.width, 1.0f / tex.height);

    vector2 one = vector2(1.0f, 1.0f);

    // fractionalOffset
    vector2 f = pixCoord - pixCenter;
    vector2 f2 = f * f;
    vector2 f3 = f2 * f;

    vector2 omf2 = (one - f) * (one - f);
    vector2 omf3 = omf2 * (one - f);
    float sixth = (1.0f / 6.0f);
    vector2 w0 = sixth * omf3;
    vector2 w1 = ((4.0f / 6.0f) * one + 0.5f * f3 - f2);
    vector2 w3 = sixth * f3;
    vector2 w2 = one - w0 - w1 - w3;

    vector2 s0 = w0 + w1;
    vector2 s1 = w2 + w3;

    vector2 f0 = w1 / (w0 + w1);
    vector2 f1 = w3 / (w2 + w3);

    vector2 t0 = (pixCenter - one + f0) * iDim;
    vector2 t1 = (pixCenter + one + f1) * iDim;

    auto T = tex.d_texObject;
    // and sample and blend
    return vector4(tex2D<float4>(T, t0.x, t0.y)) * s0.x * s0.y + vector4(tex2D<float4>(T, t1.x, t0.y)) * s1.x * s0.y +
           vector4(tex2D<float4>(T, t0.x, t1.y)) * s0.x * s1.y + vector4(tex2D<float4>(T, t1.x, t1.y)) * s1.x * s1.y;
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

CUDA_KERNEL void TransformFoveatedSamplesToCameraSpaceKernel(const matrix3x3 eyeSpaceToSampleSpaceMatrix,
                                                             const matrix3x3 eyeSpaceToCameraSpace,
                                                             const FloatRect cullRect,
                                                             const DirectionalBeam* eyeSpaceSamples,
                                                             CameraBeams cameraBeams,
                                                             int* remap,
                                                             const uint32_t sampleCount) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sampleCount) {
        DirectionalBeam eyeBeam = eyeSpaceSamples[index];
        vector2 c = {CUDA_INF, CUDA_INF};
        DirectionalBeam cameraBeam = eyeSpaceToCameraSpace * eyeBeam;
        if (cameraBeam.centerRay.z < 0.0f) {
            vector2 s = directionToSampleSpaceSample(eyeSpaceToSampleSpaceMatrix, eyeBeam.centerRay);
            if (rectContains(cullRect, s)) {
                remap[index] = (int)index;
            }
        }
        cameraBeams.directionalBeams[index] = cameraBeam;
    }
}

void TransformFoveatedSamplesToCameraSpace(const matrix3x3& eyeSpaceToSampleSpaceMatrix,
                                           const matrix3x3& eyeSpaceToCameraSpaceMatrix,
                                           const FloatRect& sampleBounds,
                                           const DirectionalBeam* d_eyeBeams,
                                           CameraBeams cameraBeams,
                                           int* d_remap,
                                           const uint32_t sampleCount,
                                           cudaStream_t stream) {
    KernelDim dim = KernelDim(sampleCount, CUDA_GROUP_SIZE);
    TransformFoveatedSamplesToCameraSpaceKernel<<<dim.grid, dim.block, 0, stream>>>(
        eyeSpaceToSampleSpaceMatrix, eyeSpaceToCameraSpaceMatrix, sampleBounds, d_eyeBeams, cameraBeams, d_remap,
        sampleCount);
}

struct EccentricityToTexCoordMapping {
    EccentricityMap eccentricityMap;
    float texMapSize;
    float invTexMapSize;
    float invMaxEccentricity;
};

void GPUCamera::getEccentricityMap(EccentricityToTexCoordMapping& map) const {
    map.eccentricityMap = eccentricityMap;
    map.texMapSize = (float)polarTextures.raw.height;
    map.invTexMapSize = 1.0f / polarTextures.raw.height;
    map.invMaxEccentricity = 1.0f / maxEccentricityRadians;
}

// eccentricity is in the range [0,maxEccentricityRadians]
CUDA_DEVICE float eccentricityToTexCoord(float eccentricity, EccentricityToTexCoordMapping eToTexMap) {
    return (eToTexMap.eccentricityMap.applyInverse(eccentricity) + 0.5f) * eToTexMap.invTexMapSize;
}

CUDA_DEVICE vector2 getNormalizedCoord(int x, int y, int width, int height) {
    return vector2(((float)x + 0.5f) / (float)width, ((float)y + 0.5f) / (float)height);
}

// Aligned along z axis
CUDA_DEVICE vector3 angularEyeCoordToDirection(float theta, float e) {
    float z = -cosf(e);
    float xyLength = sqrtf(1.0f - z * z);
    vector2 xy = vector2(cosf(-theta), sinf(-theta)) * xyLength;
    return {xy.x, xy.y, z};
}
CUDA_DEVICE void eyeSpaceDirectionToAngularEyeCoord(vector3 dir, float& theta, float& eccentricity) {
    eccentricity = acosf(-dir.z);
    // Angle of rotation about z, measured from x
    theta = -atan2f(dir.y, dir.x);
}

CUDA_DEVICE void polarTextureCoordToAngularEyeCoord(vector2 coord,
                                                    EccentricityToTexCoordMapping eToTexMap,
                                                    float& theta,
                                                    float& eccentricity) {
    eccentricity = eToTexMap.eccentricityMap.apply(coord.y * eToTexMap.texMapSize - 0.5f);
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

CUDA_DEVICE vector4 sqrt(vector4 v) {
    return vector4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}

CUDA_DEVICE vector4 clampToNeighborhood(vector4 oldValue,
                                        GPUCamera::PolarTextures polarTex,
                                        vector2 coord,
                                        TemporalFilterSettings settings) {
    vector4 m_1 = vector4(tex2D<float4>(polarTex.moment1.d_texObject, coord.x, coord.y));
    vector4 m_2 = vector4(tex2D<float4>(polarTex.moment2.d_texObject, coord.x, coord.y));
    vector4 stdDev = sqrt(m_2 - (m_1 * m_1));
    // Arbitrary
    float scaleFactor = settings.stddevMultiplier;
    vector4 minC = m_1 - (stdDev * scaleFactor);
    vector4 maxC = m_1 + (stdDev * scaleFactor);
    return clamp(oldValue, minC, maxC);
}

template <PixelFormat PixelFormat>
CUDA_KERNEL void ComputeMoments(GPUCamera::PolarTextures polarTex) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < polarTex.raw.width && j < polarTex.raw.height) {
        vector4 m_1, m_2;
        vector2 invDim = vector2(1.0f / polarTex.raw.width, 1.0f / polarTex.raw.height);
        vector2 coord = vector2(invDim.x * i, invDim.y * j);
        computeMoments3x3Window(polarTex.raw.d_texObject, coord, invDim, m_1, m_2);
        writeSurface<PixelFormat>(m_1, polarTex.moment1, i, j);
        writeSurface<PixelFormat>(m_2, polarTex.moment2, i, j);
    }
}

template <PixelFormat PixelFormat>
CUDA_KERNEL void FoveatedPolarToScreenSpaceKernel(GPUCamera::PolarTextures polarTex,
                                                  Texture2D resultTexture,
                                                  GPUImage resultImage,
                                                  matrix3x3 sampleSpaceToEyeSpaceMatrix,
                                                  EccentricityToTexCoordMapping eToTexMap,
                                                  Texture2D previousResultTexture,
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

        /** Display full mapping, for debugging
        theta = normalizedCoord.x * 2.0f * Pi;
        eccentricity = normalizedCoord.y * eToTexMap.invMaxEccentricity;
        */

        vector2 coord = angularEyeCoordToPolarTextureCoord(theta, eccentricity, eToTexMap);

        vector4 newValue = bicubicFast(polarTex.raw, coord);
        vector4 result = newValue;
        vector4 surfaceResult = result;

        float tValue = tex2D<float>(polarTex.depth.d_texObject, coord.x, coord.y);
        if (tValue < CUDA_INF) {
            vector3 currentEyePosition = angularEyeCoordToDirection(theta, eccentricity) * tValue;

            vector4 prevSamplePosition = eyeSpaceToPreviousSampleSpaceMatrix * vector4(currentEyePosition, 1.0f);
            vector2 oldTexCoord = vector2(prevSamplePosition.x, prevSamplePosition.y) * (1.0f / prevSamplePosition.z);

            float alpha = settings.alpha;
            vector4 oldValue = newValue;
            if (oldTexCoord.x > 0 && oldTexCoord.y > 0 && oldTexCoord.x < 1 && oldTexCoord.y < 1 && tValue > 0) {
                oldValue = vector4(tex2D<float4>(previousResultTexture.d_texObject, oldTexCoord.x, oldTexCoord.y));
            }


            vector4 clampedOldValue = clampToNeighborhood(oldValue, polarTex, coord, settings);

            // Make alpha settings be dependent on eccentricity. Make it higher in fovea and lower toward periphery
            float normalizedE = eccentricity * eToTexMap.invMaxEccentricity;
            float mn = 0.2f, mx = 0.35f;
            float t = clamp((normalizedE - mn) / (mx - mn), 0.f, 1.f);
            alpha = lerp(0.5f, alpha, t);

            // Heuristic hack! Turn down TAA clamping in the periphery
            normalizedE = sqrtf(sqrtf(sqrtf(sqrtf(normalizedE))));
            clampedOldValue.x = lerp(clampedOldValue.x, oldValue.x, normalizedE);
            clampedOldValue.y = lerp(clampedOldValue.y, oldValue.y, normalizedE);
            clampedOldValue.z = lerp(clampedOldValue.z, oldValue.z, normalizedE);
            clampedOldValue.w = lerp(clampedOldValue.w, oldValue.w, normalizedE);

            surfaceResult = alpha * newValue + (1.0f - alpha) * clampedOldValue;
            result = surfaceResult;
        }


        if (PixelFormat == PixelFormat::RGBA32F) {
            vector4* output = (vector4*)resultImage.data();
            output[resultImage.stride() * j + i] = result;
        } else {
            uint32_t* output = (uint32_t*)resultImage.data();
            output[resultImage.stride() * j + i] = ToColor4Unorm8SRgb(result);
        }
        writeSurface<PixelFormat>(surfaceResult, resultTexture, i, j);
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

// The reliance on eccentricity is a pure guess, a better implementation would make this more principled or
// at least try and obtain the formula used (but not published) in
// https://research.nvidia.com/sites/default/files/publications/supplementary.pdf
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

        int filterRadius = 5;
        vector4 valueSum = vector4(0.0f, 0.0f, 0.0f, 0.0f);
        float weightSum = 0.0f;
        for (int R = -filterRadius; R <= filterRadius; ++R) {
            vector2i tapLoc(clamp((int)i + R * step.x, (int)0, (int)output.width - 1),
                            clamp((int)j + R * step.y, (int)0, (int)output.height - 1));
            float normDist = fabsf(float(R)) / float(filterRadius + 0.1);
            float weight = powf(1.0f - normDist, sqrtf(eccentricity));
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
                                            matrix3x3 sampleSpaceToEyeSpaceMatrix,
                                            EccentricityToTexCoordMapping eToTexMap) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < resultImage.width() && j < resultImage.height()) {
        // TODO: compute sigma
        float eccentricity = getEccentricity(i, j, unfilteredTexture, sampleSpaceToEyeSpaceMatrix);
        float sigma = 8.0f;

        float t = eccentricity * eToTexMap.invMaxEccentricity;

        sigma *= max(0.001f, clamp(t * t, 0.f, 1.f));

        vector4 pix = texelFetch(unfilteredTexture, i, j);
        vector4 pmean = texelFetch(filteredTexture, i, j);

        vector4 result = enhanceContrast(pix, pmean, sigma, settings.f_e);

        uint32_t* output = (uint32_t*)resultImage.data();
        output[resultImage.stride() * j + i] = ToColor4Unorm8SRgb(result);
    }
}

void GPUCamera::foveatedPolarToScreenSpace(const matrix4x4& eyeToEyePrevious,
                                           const matrix3x3& eyePreviousToSamplePrevious,
                                           const matrix3x3& sampleToEye) {
    KernelDim dim = KernelDim(resultImage.width(), resultImage.height(), CUDA_GROUP_WIDTH, CUDA_GROUP_HEIGHT);
    KernelDim polDim =
        KernelDim(polarTextures.raw.width, polarTextures.raw.height, CUDA_GROUP_WIDTH, CUDA_GROUP_HEIGHT);
    matrix4x4 eyeToSamplePrevious = matrix4x4(eyePreviousToSamplePrevious) * eyeToEyePrevious;

    EccentricityToTexCoordMapping eToTexMap;
    getEccentricityMap(eToTexMap);
    ComputeMoments<PixelFormat::RGBA16><<<polDim.grid, polDim.block, 0, stream>>>(polarTextures);
    switch (outputModeToPixelFormat(outputMode)) {
        case PixelFormat::RGBA32F:
            FoveatedPolarToScreenSpaceKernel<PixelFormat::RGBA32F><<<dim.grid, dim.block, 0, stream>>>(
                polarTextures, resultTexture, resultImage, sampleToEye, eToTexMap, previousResultTexture,
                eyeToSamplePrevious, temporalFilterSettings);
            break;
        case PixelFormat::RGBA8_SRGB:
            FoveatedPolarToScreenSpaceKernel<PixelFormat::RGBA8_SRGB><<<dim.grid, dim.block, 0, stream>>>(
                polarTextures, resultTexture, resultImage, sampleToEye, eToTexMap, previousResultTexture,
                eyeToSamplePrevious, temporalFilterSettings);
            break;
        default:
            assert(false);
    }

    if (contrastEnhancementSettings.enable) {
        assert(outputModeToPixelFormat(outputMode) == PixelFormat::RGBA8_SRGB);
        SeparableFilterUsingEccentricity<PixelFormat::RGBA8_SRGB>
            <<<dim.grid, dim.block, 0, stream>>>(contrastEnhancementBuffers.horizontallyFiltered, resultTexture, {0, 1},
                                                 contrastEnhancementSettings, sampleToEye);
        SeparableFilterUsingEccentricity<PixelFormat::RGBA8_SRGB><<<dim.grid, dim.block, 0, stream>>>(
            contrastEnhancementBuffers.fullyFiltered, contrastEnhancementBuffers.horizontallyFiltered, {1, 0},
            contrastEnhancementSettings, sampleToEye);
        FinishConstrastEnhancement<PixelFormat::RGBA8_SRGB>
            <<<dim.grid, dim.block, 0, stream>>>(resultImage, resultTexture, contrastEnhancementBuffers.fullyFiltered,
                                                 contrastEnhancementSettings, sampleToEye, eToTexMap);
    }

    std::swap(previousResultTexture, resultTexture);
}

void GPUCamera::updateEyeSpaceFoveatedSamples(const ArrayView<DirectionalBeam> eyeBeams) {
    d_foveatedEyeDirectionalSamples = GPUBuffer<DirectionalBeam>(eyeBeams.cbegin(), eyeBeams.cend());

    // Allocate and calculate eye-space frusta
    uint32_t blockCount = ((uint32_t)eyeBeams.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_foveatedEyeSpaceTileFrusta = GPUBuffer<SimpleRayFrustum>(blockCount * TILES_PER_BLOCK);
    d_foveatedEyeSpaceBlockFrusta = GPUBuffer<SimpleRayFrustum>(blockCount);
    d_foveatedWorldSpaceTileFrusta = GPUBuffer<SimpleRayFrustum>(blockCount * TILES_PER_BLOCK);
    d_foveatedWorldSpaceBlockFrusta = GPUBuffer<SimpleRayFrustum>(blockCount);

    ComputeEyeSpaceFrusta(d_foveatedEyeDirectionalSamples, d_foveatedEyeSpaceTileFrusta, d_foveatedEyeSpaceBlockFrusta);

    safeCudaEventDestroy(transferTileToCPUEvent);
    cutilSafeCall(cudaEventCreateWithFlags(&transferTileToCPUEvent, cudaEventDisableTiming));

    safeCudaFreeHost(foveatedWorldSpaceTileFrustaPinned);
    safeCudaFreeHost(foveatedWorldSpaceBlockFrustaPinned);

    cutilSafeCall(cudaMallocHost((void**)&foveatedWorldSpaceTileFrustaPinned,
                                 sizeof(SimpleRayFrustum) * blockCount * TILES_PER_BLOCK));
    cutilSafeCall(cudaMallocHost((void**)&foveatedWorldSpaceBlockFrustaPinned, sizeof(SimpleRayFrustum) * blockCount));
}

Plane operator*(matrix4x4 M, Plane p) {
    vector4 O = vector4(p.normal * p.dist, 1.0f);
    O = M * O;
    vector3 N = vector3(transpose(invert(M)) * vector4(p.normal, 0));
    return Plane{N, dot(vector3(O), N)};
}

void GPUCamera::updatePerFrameFoveatedData(const FloatRect& sampleBounds,
                                           const matrix3x3& cameraToSample,
                                           const matrix3x3& eyeToCamera,
                                           const matrix4x4& eyeToWorld) {
    validSampleCount = uint32_t(d_foveatedEyeDirectionalSamples.size());
    CameraBeams cameraBeams(*this);
    uint32_t tileCount = uint32_t(d_foveatedWorldSpaceTileFrusta.size());
    uint32_t blockCount = uint32_t(d_foveatedWorldSpaceBlockFrusta.size());


    matrix3x3 eyeToSample = cameraToSample * eyeToCamera;
    TransformFoveatedSamplesToCameraSpace(eyeToSample, eyeToCamera, sampleBounds, d_foveatedEyeDirectionalSamples,
                                          cameraBeams, d_sampleRemap, validSampleCount, stream);

    auto sampleToCamera = invert(cameraToSample);

    auto U = sampleBounds.upper;
    auto L = sampleBounds.lower;

    vector2 sampleDirs[4] = {{U.x, U.y}, {L.x, U.y}, {L.x, L.y}, {U.x, L.y}};

    const float EPSILON = -0.01f;
    FourPlanes cullPlanes;
    for (int i = 0; i < 4; ++i) {
        vector3 dir0 = sampleToCamera * vector3(sampleDirs[i], 1.0f);
        vector3 dir1 = sampleToCamera * vector3(sampleDirs[(i + 1) % 4], 1.0f);
        Plane eyeSpacePlane;
        eyeSpacePlane.normal = invert(eyeToCamera) * normalize(cross(dir1, dir0));
        eyeSpacePlane.dist = EPSILON;
        cullPlanes.data[i] = eyeToWorld * eyeSpacePlane;
    }

    CalculateWorldSpaceFrusta(d_foveatedWorldSpaceBlockFrusta, d_foveatedWorldSpaceTileFrusta,
                              d_foveatedEyeSpaceBlockFrusta, d_foveatedEyeSpaceTileFrusta, eyeToWorld, cullPlanes,
                              blockCount, tileCount, stream);
    // Queue the copy back
    d_foveatedWorldSpaceBlockFrusta.readbackAsync(foveatedWorldSpaceBlockFrustaPinned, stream);
    d_foveatedWorldSpaceTileFrusta.readbackAsync(foveatedWorldSpaceTileFrustaPinned, stream);

    cutilSafeCall(cudaEventRecord(transferTileToCPUEvent, stream));
}

} // namespace hvvr
