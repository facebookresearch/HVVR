#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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

// TODO(anankervis): merge into Raycaster
class GPUSceneState {
public:
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
    // if an updated BVH is available on the GPU, returns true and copies the GPU data to the CPU pointer
    // assumes size of dstNodes matches the number of nodes last passed into UpdateSceneGeometry
    bool fetchUpdatedBVH(BVHNode* dstNodes);

    // if this is called, animate must also be called afterward to populate the transforms
    void setGeometry(const Raycaster& raycaster);
    void animate(const matrix4x4* modelToWorld, size_t modelToWorldCount);

    void updateLighting(const Raycaster& raycaster);
    void updateMaterials(SimpleMaterial* _materials, size_t materialCount);

protected:
};

} // namespace hvvr
