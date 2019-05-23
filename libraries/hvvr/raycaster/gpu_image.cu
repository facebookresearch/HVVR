/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_util.h"
#include "gpu_image.h"

namespace hvvr {

void GPUImage::update(uint32_t imageWidth, uint32_t imageHeight, uint32_t imageStride, PixelFormat imageFormat) {
    if (imageWidth != m_height || imageHeight != m_height || imageStride != m_stride || imageFormat != m_format) {
        m_width = imageWidth;
        m_height = imageHeight;
        m_stride = imageStride;
        m_format = imageFormat;
        assert(m_backingGraphicsResource == nullptr);
        if (d_data) {
            cutilSafeCall(cudaFree(d_data));
        }
        cutilSafeCall(cudaMalloc(&d_data, sizeInMemory()));
    }
}

void GPUImage::updateFromLinearGraphicsResource(cudaGraphicsResource_t resource, size_t size, PixelFormat imageFormat) {
    assert(size <= uint32_t(-1));
    size_t maxSize = 0;
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer(((void**)(&d_data)), &maxSize, resource));
    assert(maxSize / bytesPerPixel() >= size);
    m_format = imageFormat;
    m_height = 1;
    m_width = m_stride = uint32_t(size);
    m_backingGraphicsResource = resource;
}

void GPUImage::reset() {
    m_width = 0;
    m_height = 0;
    m_stride = 0;
    m_format = PixelFormat::RGBA8_SRGB;
    if ((!m_backingGraphicsResource) && d_data) {
        cutilSafeCall(cudaFree(d_data));
    }
    d_data = nullptr;
}

} // namespace hvvr
