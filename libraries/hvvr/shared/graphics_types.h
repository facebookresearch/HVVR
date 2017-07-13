#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "debug.h"
#include "util.h"
#include "vector_math.h"

#include <string.h>
#include <string>
#include <vector>

namespace hvvr {
//==============================================================================
// Structures used by various graphics types.
//==============================================================================

struct Rect {
    vector2i lower, upper;
};
struct FloatRect {
    vector2 lower, upper;

    FloatRect() {}
    FloatRect(const vector2& _lower, const vector2& _upper) : lower(_lower), upper(_upper) {}
};

// A simple image view: Does not own any memory.  Can wrap an arbitrary region in a 2-D buffer.
struct ImageViewR8G8B8A8 {
    ImageViewR8G8B8A8() = default;
    ImageViewR8G8B8A8(uint32_t* data, uint32_t width, uint32_t height, size_t stride)
        : data(data), width(width), height(height), stride(stride) {}
    __forceinline unsigned* operator[](size_t row) const {
        return data + row * stride;
    }

    uint32_t* data = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t stride = 0;
};

enum class PixelFormat { RGBA8_SRGB, RGBA32F };

// A simple image resource handle: Does not own any memory.
// Can wrap an arbitrary region in a 2-D buffer.
// Can contain different pixel formats, either in regular memory, cuda memory,
// DX Buffer, or GL Buffer
// Since the data can be on the GPU, does not allow direct access.
struct ImageResourceDescriptor {
    enum class MemoryType { CPU_MEMORY, CUDA_MEMORY, DX_TEXTURE, OPENGL_TEXTURE };
    ImageResourceDescriptor() = default;
    explicit ImageResourceDescriptor(const ImageViewR8G8B8A8& view)
        : data(view.data), width(view.width), height(view.height), stride(view.stride) {}

    bool isHardwareRenderTarget() const {
        return !(memoryType == MemoryType::CPU_MEMORY);
    }

    bool operator==(const ImageResourceDescriptor& other) const {
        return (pixelFormat == other.pixelFormat) && (memoryType == other.memoryType) && (data == other.data) &&
               (width == other.width) && (height == other.height) && (stride == other.stride);
    }
    bool operator!=(const ImageResourceDescriptor& other) const {
        return !(operator==(other));
    }

    PixelFormat pixelFormat = PixelFormat::RGBA8_SRGB;
    MemoryType memoryType = MemoryType::CPU_MEMORY;
    void* data = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t stride = 0;
};

// Shading vertex (32 bytes)
struct ShadingVertex {
    vector3 pos;
    vector4h normal;
    vector2 uv;
    uint32_t pad0;
};

// we need a new struct, because some ShadingVertex members are compressed
struct InterpolatedVertex {
    vector3 pos;
    vector3 normal;
    vector2 uv;
};

// Skin weights (8 bytes)
struct ALIGN(4) SkinWeights {
    static const size_t MAX_WEIGHTS = 4;
    uint8_t indices[MAX_WEIGHTS]; // Bone indices
    uint8_t weights[MAX_WEIGHTS]; // Bone weights
};

struct SkeletonNode {
    SkeletonNode() = default;
    SkeletonNode(const char* name_, size_t parentIndex, const transform& parentFromNode)
        : parentFromNode(parentFromNode), parentIndex(parentIndex) {
        memset(name, 0, sizeof(name));

#ifdef _WIN32
        if (strcpy_s(name, name_))
            fail("Node name is too long.");
#else
        if (strncmp(name, name_, sizeof(name)) >= (int)sizeof(name))
            fail("Node name is too long.");
        strcpy(name, name_);
#endif
    }

    transform parentFromNode = transform::identity(); // transform of the node
    size_t parentIndex = 0;                           // Index of parent node
    char name[96 - sizeof(size_t)];                   // Node name
};
static_assert(sizeof(SkeletonNode) == 0x80, "I like round numbers.");

struct TopologyNode {
    __forceinline uint32_t isLeaf(size_t i) const {
        return leafMask & (1u << i);
    }
    __forceinline uint32_t getFirstTriangleIndex(size_t i) const {
        return data[i];
    }
    __forceinline uint32_t getBoundTriangleIndex(size_t i) const {
        return data[1 + i];
    }
    __forceinline uint32_t getChildOffset(size_t i) const {
        return data[1 + i];
    }
    __forceinline void setFirstTriangleIndex(size_t i, uint32_t value) {
        data[i] = value;
    }
    __forceinline void setBoundTriangleIndex(size_t i, uint32_t value) {
        data[1 + i] = value;
    }
    __forceinline void setChildOffset(size_t i, uint32_t value) {
        data[1 + i] = value;
    }

    uint32_t leafMask;
    uint32_t data[7];
};

struct KeyFrame {
    float Time;
    vector4 Trans;
    quaternion Rot;
};

struct AnimTrackNode {
    std::vector<KeyFrame> keyFrames;
};

struct AnimationNode {
    std::string Name;
    float Duration; // animation duration in seconds
    std::vector<AnimTrackNode> tracks;
};

// Parameterization of a thin lens.
struct ThinLens {
    float radius;
    float focalDistance;
};

struct SimpleRay {
    vector3 origin;
    vector3 direction;
};

// A simple encoding of a frustum (sans far plane) as rays with four separate origin/directions.
// Easy to transform.
struct SimpleRayFrustum {
    vector3 origins[4];
    vector3 directions[4];
};

// A precomputed triangle, optimized for intersection.
struct PrecomputedTriangleIntersect {
    vector3 v0;
    vector3 edge0;
    vector3 edge1;
};
struct PrecomputedTriangleShade {
    uint32_t indices[3];
    uint32_t material;
};

} // namespace hvvr
