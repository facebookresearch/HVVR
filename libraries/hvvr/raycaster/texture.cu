/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_decl.h"
#include "cuda_util.h"
#include "material.h"
#include "texture.h"
#include "texture_internal.h"
#include "vector_math.h"

#include <vector>


namespace hvvr {

struct CudaFormatDescriptor {
    uint32_t r = 0, g = 0, b = 0, a = 0;
    cudaChannelFormatKind channelType = cudaChannelFormatKindNone;
    cudaTextureReadMode readMode = cudaReadModeElementType;
    bool sRGB = false;
    uint32_t elementSize = 0;
    CudaFormatDescriptor() {}
    CudaFormatDescriptor(uint32_t r,
                         uint32_t g,
                         uint32_t b,
                         uint32_t a,
                         cudaChannelFormatKind channelType,
                         cudaTextureReadMode readMode,
                         bool sRGB,
                         uint32_t elementSize)
        : r(r), g(g), b(b), a(a), channelType(channelType), readMode(readMode), sRGB(sRGB), elementSize(elementSize) {}
};

static CudaFormatDescriptor formatToDescriptor(TextureFormat format) {
    switch (format) {
        case TextureFormat::r8g8b8a8_unorm_srgb:
            return {8u, 8u, 8u, 8u, cudaChannelFormatKindUnsigned, cudaReadModeNormalizedFloat, true, 4};
        case TextureFormat::r8g8b8a8_unorm:
            return {8u, 8u, 8u, 8u, cudaChannelFormatKindUnsigned, cudaReadModeNormalizedFloat, false, 4};
		case TextureFormat::r16g16b16a16_unorm:
			return{ 16u, 16u, 16u, 16u, cudaChannelFormatKindUnsigned, cudaReadModeNormalizedFloat, false, 4 };
        case TextureFormat::r32g32b32a32_float:
            return {32u, 32u, 32u, 32u, cudaChannelFormatKindFloat, cudaReadModeElementType, false, 16};
        case TextureFormat::r16g16b16a16_float:
            return {16u, 16u, 16u, 16u, cudaChannelFormatKindFloat, cudaReadModeElementType, false, 8};
        case TextureFormat::r11g11b10_float:
            return {11u, 11u, 10u, 0u, cudaChannelFormatKindFloat, cudaReadModeElementType, false, 4};
        case TextureFormat::r32_float:
            return {32u, 0u, 0u, 0u, cudaChannelFormatKindFloat, cudaReadModeElementType, false, 4};
        default:
            printf("Unhandled texture format\n");
            assert(false);
    }
    return CudaFormatDescriptor();
}

Texture::Texture(const TextureData& textureData) {
    _textureID = CreateTexture(textureData);
}

// TODO(anankervis):
Texture::~Texture() {}


cudaTextureObject_t* gDeviceTextureArray;
Texture2D gTextureAtlas[SimpleMaterial::maxTextureCount] = {};
static uint32_t gTextureCount = 0;

CUDA_DEVICE uchar4 to_uchar4(vector4 vec) {
    return make_uchar4((uint8_t)vec.x, (uint8_t)vec.y, (uint8_t)vec.z, (uint8_t)vec.w);
}

CUDA_KERNEL void d_mipmap(cudaSurfaceObject_t mipOutput,
                          cudaTextureObject_t mipInput,
                          uint32_t imageW,
                          uint32_t imageH) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0 / float(imageW);
    float py = 1.0 / float(imageH);

    if ((x < imageW) && (y < imageH)) {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        vector4 color = vector4(tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)) +
                        vector4(tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)) +
                        vector4(tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)) +
                        vector4(tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py));

        color /= 4.0f;
        color *= 255.0f;
        color = min(color, 255.0f);

        surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
    }
}

static void generateMipMaps(cudaMipmappedArray_t mipmapArray, uint32_t width, uint32_t height) {
#ifdef SHOW_MIPMAPS
    cudaArray_t levelFirst;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFirst, mipmapArray, 0));
#endif

    uint32_t level = 0;

    while (width != 1 || height != 1) {
        width /= 2;
        width = max(uint32_t(1), width);
        height /= 2;
        height = max(uint32_t(1), height);

        cudaArray_t levelFrom;
        cutilSafeCall(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
        cudaArray_t levelTo;
        cutilSafeCall(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

        cudaExtent levelToSize;
        cutilSafeCall(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
        assert(levelToSize.width == width);
        assert(levelToSize.height == height);
        assert(levelToSize.depth == 0);

        // generate texture object for reading
        cudaTextureObject_t texInput;
        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = levelFrom;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;

        texDescr.readMode = cudaReadModeNormalizedFloat;

        cutilSafeCall(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

        // generate surface object for writing

        cudaSurfaceObject_t surfOutput;
        cudaResourceDesc surfRes;
        memset(&surfRes, 0, sizeof(cudaResourceDesc));
        surfRes.resType = cudaResourceTypeArray;
        surfRes.res.array.array = levelTo;

        cutilSafeCall(cudaCreateSurfaceObject(&surfOutput, &surfRes));

        // run mipmap kernel
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);
        d_mipmap<<<gridSize, blockSize>>>(surfOutput, texInput, width, height);

        cutilSafeCall(cudaDeviceSynchronize());
        cutilSafeCall(cudaGetLastError());

        cutilSafeCall(cudaDestroySurfaceObject(surfOutput));
        cutilSafeCall(cudaDestroyTextureObject(texInput));

#ifdef SHOW_MIPMAPS
        // we blit the current mipmap back into first level
        cudaMemcpy3DParms copyParams = {0};
        copyParams.dstArray = levelFirst;
        copyParams.srcArray = levelTo;
        copyParams.extent = make_cudaExtent(width, height, 1);
        copyParams.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
#endif

        level++;
    }
}

uint32_t getMipMapLevels(uint32_t width, uint32_t height, uint32_t depth) {
    uint32_t sz = max(max(width, height), depth);

    uint32_t levels = 0;
    while (sz) {
        sz /= 2;
        levels++;
    }

    return levels;
}

// CPU allocates resources address
uint32_t CreateTexture(const TextureData& textureData) {
    uint32_t depth = 0;

    assert(gTextureCount < SimpleMaterial::maxTextureCount - 1); // reserve the last index for SimpleMaterial::badTextureIndex

    if (gTextureCount == 0) {
        cudaMalloc((void**)(&gDeviceTextureArray), sizeof(cudaTextureObject_t) * SimpleMaterial::maxTextureCount);
    }

    CudaFormatDescriptor desc = formatToDescriptor(textureData.format);

    Texture2D tex;
    tex.width = textureData.width;
    tex.height = textureData.height;
    tex.elementSize = desc.elementSize;
    tex.hasMipMaps = true;
    tex.format = textureData.format;

    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc(desc.r, desc.g, desc.b, desc.a, desc.channelType);
    cudaExtent extents = {textureData.width, textureData.height, depth};
    uint32_t levels = 0;
    if (tex.hasMipMaps) {
        // how many mipmaps we need
        levels = getMipMapLevels(textureData.width, textureData.height, depth);
        cutilSafeCall(cudaMallocMipmappedArray(&tex.d_rawMipMappedMemory, &chanDesc, extents, levels));

        // upload level 0
        cutilSafeCall(cudaGetMipmappedArrayLevel(&tex.d_rawMemory, tex.d_rawMipMappedMemory, 0));
    } else {
        // Create buffer for cuda write
        cutilSafeCall(cudaMallocArray(&tex.d_rawMemory, &chanDesc, textureData.width, textureData.height));
    }

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = desc.readMode;
    texDesc.sRGB = desc.sRGB;
    texDesc.normalizedCoords = true;
    texDesc.maxAnisotropy = 8;

    printf("width: %u, height: %u, stride: %u, elementSize: %u\n", textureData.width, textureData.height,
           textureData.strideElements, desc.elementSize);
    cutilSafeCall(cudaMemcpy2DToArray(tex.d_rawMemory, 0, 0, textureData.data, textureData.strideElements * desc.elementSize,
                                      textureData.width * desc.elementSize, textureData.height,
                                      cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    if (tex.hasMipMaps) {
        generateMipMaps(tex.d_rawMipMappedMemory, textureData.width, textureData.height);

        resDesc.resType = cudaResourceTypeMipmappedArray;
        resDesc.res.mipmap.mipmap = tex.d_rawMipMappedMemory;

        texDesc.mipmapFilterMode = cudaFilterModeLinear;
        texDesc.maxMipmapLevelClamp = float(levels - 1);
    } else {
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = tex.d_rawMemory;
    }

    // Create Texture Object
    cutilSafeCall(cudaCreateTextureObject(&tex.d_texObject, &resDesc, &texDesc, 0));
    cutilSafeCall(cudaMemcpy(&gDeviceTextureArray[gTextureCount], &tex.d_texObject, sizeof(cudaTextureObject_t),
                             cudaMemcpyHostToDevice));

    gTextureAtlas[gTextureCount] = tex;
    ++gTextureCount;
    return gTextureCount - 1;
}

void DestroyAllTextures() {
    for (uint32_t i = 0; i < gTextureCount; ++i) {
        cutilSafeCall(cudaFreeArray(gTextureAtlas[i].d_rawMemory));
        cutilSafeCall(cudaDestroyTextureObject(gTextureAtlas[i].d_texObject));
    }
    cutilSafeCall(cudaFree(gDeviceTextureArray));
    gTextureCount = 0;
}

Texture2D createEmptyTexture(uint32_t width,
                             uint32_t height,
                             TextureFormat format,
                             cudaTextureAddressMode xWrapMode,
                             cudaTextureAddressMode yWrapMode,
                             bool linearFilter) {
    CudaFormatDescriptor desc = formatToDescriptor(format);

    Texture2D tex;
    tex.width = width;
    tex.height = height;
    tex.elementSize = desc.elementSize;
    tex.hasMipMaps = false;
    tex.format = format;

    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc(desc.r, desc.g, desc.b, desc.a, desc.channelType);
    // Create buffer for cuda write
    cutilSafeCall(cudaMallocArray(&tex.d_rawMemory, &chanDesc, width, height));

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = xWrapMode;
    texDesc.addressMode[1] = yWrapMode;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = linearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
    texDesc.readMode = desc.readMode;
    texDesc.normalizedCoords = true;
    texDesc.sRGB = desc.sRGB;

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex.d_rawMemory;

    // Create Texture Object
    cutilSafeCall(cudaCreateTextureObject(&tex.d_texObject, &resDesc, &texDesc, 0));
    // Create Surface Object
    cutilSafeCall(cudaCreateSurfaceObject(&tex.d_surfaceObject, &resDesc));

    return tex;
}

CUDA_KERNEL void ClearKernel(Texture2D tex) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < tex.width*tex.elementSize && y < tex.height) {
		surf2Dwrite<unsigned char>(0, tex.d_surfaceObject, x, y);
	}
}

void clearTexture(Texture2D tex) {
	KernelDim dim(tex.width*tex.elementSize, tex.height, 16, 8);
	ClearKernel<<<dim.grid, dim.block>>>(tex);
}

} // namespace hvvr
