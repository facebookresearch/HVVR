#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gpu_buffer.h"
#include "graphics_types.h"
#include "lighting_env.h"
#include "vector_math.h"
#include "dynamic_array.h"

struct BVHNode;

namespace hvvr {

class Raycaster;
struct SimpleMaterial;

// one node per thread, dependencies determined offline
struct SceneRefitTaskNode {
    uint32_t childIndex; // the child node
    uint32_t nodeChild; // the parent node and child index
};

struct GPUSceneState {
    GPUBuffer<PrecomputedTriangleIntersect> trianglesIntersect; // filled by GPU
    GPUBuffer<PrecomputedTriangleShade> trianglesShade;         // copied from CPU
    GPUBuffer<ShadingVertex> untransformedVertices;
    GPUBuffer<matrix4x4> _modelToWorld;
    GPUBuffer<int> vertexToBoneOffsetMapping;
    GPUBuffer<uint32_t> vertexToTransformMapping;
    GPUBuffer<ShadingVertex> worldSpaceVertices;
    GPUBuffer<SimpleMaterial> materials;

    GPUBuffer<BVHNode> nodes;
    GPUBuffer<uint32_t> nodeReady;
    GPUBuffer<uint32_t> triToNodeChild;
    GPUBuffer<SceneRefitTaskNode> refitTasksNode;
    DynamicArray<uint32_t> refitNodeGroupBoundaries;

    LightingEnvironment lightingEnvironment;

    cudaEvent_t updateEvent;
    cudaStream_t stream;

    bool updateDirty;
    bool updatedBVHAvailable;

    GPUSceneState();
    ~GPUSceneState();
    void reset();

    void update();
    bool fetchUpdatedBVH(BVHNode* dstNodes);

    void setGeometry(const Raycaster& raycaster);
    void animate(const matrix4x4* modelToWorld, size_t modelToWorldCount);
};

} // namespace hvvr
