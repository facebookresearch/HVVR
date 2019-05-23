/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gpu_camera.h"
#include "gpu_context.h"
#include "graphics_types.h"
#include "kernel_constants.h"
#include "shading_helpers.h"


namespace hvvr {

CUDA_DEVICE vector4 mergeSplitColors(const vector4* c) {
    return vector4(c[0].x, c[1].y, c[2].z, 1.0f);
}
CUDA_DEVICE uint32_t mergeSplitColors(const uint32_t* c) {
    return uint32_t((c[0] & 0xff) | (c[1] & 0xff00) | (c[2] & 0xff0000) | 0xff000000);
}

CUDA_DEVICE vector4 mergeSplitColorsPentile(const vector4* c, uint32_t x, uint32_t y) {
    return vector4(c[1].x, c[0].y, c[1].z, 1.0f);
}
CUDA_DEVICE uint32_t mergeSplitColorsPentile(const uint32_t* c, uint32_t x, uint32_t y) {
    uint32_t merged((c[1] & 0xff) | (c[0] & 0xff00) | (c[1] & 0xff0000) | 0xff000000);
    return merged;
}

// TODO(anankervis): for SplitColorSamples > 1, store each channel as a single component, instead of wasting space on
// RGBA PixelType
template <class PixelType, uint32_t SplitColorSamples>
CUDA_KERNEL void RemapKernel(PixelType* src,
                             int32_t* remap,
                             PixelType* dstImage,
                             uint32_t imageWidth,
                             uint32_t imageHeight,
                             uint32_t imageStride) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageWidth && y < imageHeight) {
        if (SplitColorSamples == 1) { // one sample per output pixel
            int32_t offset = remap[imageWidth * y + x];
            if (offset >= 0) {
                PixelType p = src[offset];
                dstImage[imageStride * y + x] = p;
            }
        } else if (SplitColorSamples == 2) { // one sample per channel, pentile
            PixelType splitColors[SplitColorSamples] = {};

            for (int channel = 0; channel < SplitColorSamples; channel++) {
                int32_t offset = remap[(imageWidth * y + x) * SplitColorSamples + channel];
                if (offset >= 0) {
                    splitColors[channel] = src[offset];
                }
            }

            PixelType outputColor = mergeSplitColorsPentile(splitColors, x, y);
            dstImage[imageStride * y + x] = outputColor;
        } else if (SplitColorSamples == 3) { // one sample per R,G,B channel
            PixelType splitColors[SplitColorSamples] = {};

            for (int channel = 0; channel < SplitColorSamples; channel++) {
                int32_t offset = remap[(imageWidth * y + x) * SplitColorSamples + channel];
                if (offset >= 0) {
                    splitColors[channel] = src[offset];
                }
            }

            PixelType outputColor = mergeSplitColors(splitColors);
            dstImage[imageStride * y + x] = outputColor;
        }
    }
}

void GPUCamera::remap() {

    KernelDim dim(resultImage.width(), resultImage.height(), CUDA_GROUP_WIDTH, CUDA_GROUP_HEIGHT);
    uint32_t* d_imageData = (uint32_t*)resultImage.data();
    switch (splitColorSamples) {
        case 1: {
            enum { SplitColorSamples = 1 };
            switch (outputModeToPixelFormat(outputMode)) {
                case PixelFormat::RGBA8_SRGB:
                    RemapKernel<uint32_t, SplitColorSamples><<<dim.grid, dim.block, 0, stream>>>(
                        d_sampleResults, d_sampleRemap, d_imageData, resultImage.width(), resultImage.height(),
                        resultImage.stride());
                    break;
                default:
                    assert(false);
            }
        } break;
        case 2: {
            enum { SplitColorSamples = 2 };
            switch (outputModeToPixelFormat(outputMode)) {
                case PixelFormat::RGBA8_SRGB:
                    RemapKernel<uint32_t, SplitColorSamples><<<dim.grid, dim.block, 0, stream>>>(
                        d_sampleResults, d_sampleRemap, d_imageData, resultImage.width(), resultImage.height(),
                        resultImage.stride());
                    break;
                default:
                    assert(false);
            }
        } break;
        case 3: {
            enum { SplitColorSamples = 3 };
            switch (outputModeToPixelFormat(outputMode)) {
                case PixelFormat::RGBA8_SRGB:
                    RemapKernel<uint32_t, SplitColorSamples><<<dim.grid, dim.block, 0, stream>>>(
                        d_sampleResults, d_sampleRemap, d_imageData, resultImage.width(), resultImage.height(),
                        resultImage.stride());
                    break;
                default:
                    assert(false);
            }
        } break;
        default:
            assert(false);
            break;
    }
}

// Switching to gather might improve performance, but this will likely never be a bottleneck
CUDA_KERNEL void RemapPolarFoveatedKernel(uint32_t* src,
                                          float* tmaxSrc,
                                          vector2ui* remap,
                                          Texture2D dstImage,
                                          Texture2D tmaxDstImage,
                                          size_t rawSampleCount) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rawSampleCount) {
        vector2ui offset = remap[i];
        vector4 p = FromColor4Unorm8SRgb(src[i]);
        float t = tmaxSrc[i];
        surf2Dwrite(ToColor4Unorm8SRgb(p), dstImage.d_surfaceObject, offset.x * sizeof(uchar4), offset.y);
        surf2Dwrite(t, tmaxDstImage.d_surfaceObject, offset.x * sizeof(float), offset.y);
    }
}

void GPUCamera::remapPolarFoveated() {
    uint32_t rawSampleCount = polarTextures.raw.width * polarTextures.raw.height;
    KernelDim dim = KernelDim(rawSampleCount, CUDA_GROUP_SIZE);

    switch (outputModeToPixelFormat(outputMode)) {
        case PixelFormat::RGBA8_SRGB:
            RemapPolarFoveatedKernel<<<dim.grid, dim.block, 0, stream>>>(d_sampleResults.data(), d_tMaxBuffer,
                                                                         d_polarRemapToPixel, polarTextures.raw,
                                                                         polarTextures.depth, rawSampleCount);
            break;
        default:
            assert(false);
    }
}

} // namespace hvvr
