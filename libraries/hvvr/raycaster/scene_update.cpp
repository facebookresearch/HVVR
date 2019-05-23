/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bvh_node.h"
#include "debug.h"
#include "gpu_context.h"
#include "gpu_scene_state.h"
#include "material.h"
#include "model.h"
#include "raycaster.h"
#include "vector_math.h"


namespace hvvr {

// convert from per-model data to global arrays
void Raycaster::buildScene() {
    uint32_t modelCount = uint32_t(_models.size());

    if (modelCount == 0) {
        _trianglesShade = std::vector<PrecomputedTriangleShade>();
        _nodes = DynamicArray<BVHNode>(1);
        _materials = std::vector<SimpleMaterial>();

        // Create a root node that points to nothing.
        _nodes[0].boxData.leafMask = 0xf;
        for (size_t i = 0; i < 5; i++)
            _nodes[0].boxData.leaf.triIndex[i] = 0;

        return;
    }

    // Count the number of objects we're going to have to allocate.
    uint32_t rootNodeCount = (modelCount + 1) / 3;
    {
        _vertexCount = 0;
        uint32_t triangleCount = 0;
        uint32_t nodeCount = rootNodeCount;
        uint32_t materialCount = 0;
        for (uint32_t modelIndex = 0; modelIndex < modelCount; modelIndex++) {
            const Model& model = *(_models[modelIndex]);
            const MeshData& meshData = model.getMesh();

            _vertexCount += uint32_t(meshData.verts.size());
            triangleCount += uint32_t(meshData.triShade.size());
            nodeCount += uint32_t(meshData.nodes.size());
            materialCount += uint32_t(meshData.materials.size());
        }

        _trianglesShade = std::vector<PrecomputedTriangleShade>(triangleCount);
        _nodes = DynamicArray<BVHNode>(nodeCount);
        _materials = std::vector<SimpleMaterial>(materialCount);
    }

    // Allocate the views.
    std::vector<uint32_t> viewOffsets(_models.size());

    uint32_t vertexOffset = 0;
    uint32_t triOffset = 0;
    uint32_t materialOffset = 0;
    uint32_t nodeOffset = rootNodeCount;
    for (uint32_t modelIndex = 0; modelIndex < modelCount; modelIndex++) {
        const Model& model = *(_models[modelIndex]);
        const MeshData& meshData = model.getMesh();

        uint32_t materialCount = uint32_t(meshData.materials.size());
        for (uint32_t n = 0; n < materialCount; n++) {
            _materials[materialOffset + n] = meshData.materials[n];
        }

        uint32_t triCount = uint32_t(meshData.triShade.size());
        for (uint32_t n = 0; n < triCount; n++) {
            PrecomputedTriangleShade& triShade = _trianglesShade[triOffset + n];
            triShade = meshData.triShade[n];
            triShade.indices[0] += vertexOffset;
            triShade.indices[1] += vertexOffset;
            triShade.indices[2] += vertexOffset;
            triShade.material += materialOffset;
        }

        // Initialize the BVH
        uint32_t nodeCount = uint32_t(meshData.nodes.size());
        viewOffsets[modelIndex] = nodeOffset;
        for (uint32_t n = 0; n < nodeCount; n++) {
            const TopologyNode& topoNode = meshData.nodes[n];
            BVHNode& node = _nodes[nodeOffset + n];

            node.boxData.leafMask = topoNode.leafMask;
            node.boxData.leaf.triIndex[0] = triOffset + topoNode.getFirstTriangleIndex(0);
            for (size_t k = 0; k < 4; k++) {
                if (topoNode.isLeaf(k))
                    node.boxData.leaf.triIndex[1 + k] = triOffset + topoNode.getBoundTriangleIndex(k);
                else
                    node.boxData.children.offset[k] = topoNode.getChildOffset(k);
            }
        }

        vertexOffset += uint32_t(meshData.verts.size());
        triOffset += triCount;
        materialOffset += materialCount;
        nodeOffset += nodeCount;
    }

    if (rootNodeCount > 0) {
        // Initialize the root nodes to link the sub-trees together.
        // We use a linear vector of child pointers initially because of the ease of access.
        std::vector<BVHNode*> children(rootNodeCount * 4);
        auto nextChild = children.data();

        // Set the extra slots to nullptr (these will be empty leaves).
        for (size_t i = children.size() - _models.size() - (rootNodeCount - 1); i != 0; --i)
            *nextChild++ = nullptr;

        // Set the pointers to the fixup nodes.
        for (size_t i = 1, e = rootNodeCount; i < e; ++i)
            *nextChild++ = _nodes.data() + i;

        // Set the pointers to the entry points.
        for (auto viewOffset : viewOffsets)
            *nextChild++ = _nodes.data() + viewOffset;
        if (nextChild != children.data() + children.size())
            fail("Didn't update all of the children.");

        // Convert the pointers in the children vector into offset in the actual fixup nodes.
        for (size_t i = 0; i < rootNodeCount; i++) {
            _nodes[i].boxData.leaf.triIndex[0] = 0;
            _nodes[i].boxData.leafMask = 0;
            for (size_t k = 0; k < 4; k++) {
                if (children[4 * i + k] == nullptr) {
                    _nodes[i].boxData.leaf.triIndex[1 + k] = 0;
                    _nodes[i].boxData.leafMask |= 1 << k;
                    continue;
                }
                _nodes[i].boxData.children.offset[k] = uint32_t(children[4 * i + k] - (_nodes.data() + i));
            }
        }
    }
}

void Raycaster::uploadScene() {
    GPUSceneState& gpuSceneState = _gpuContext->sceneState;

    gpuSceneState.updateMaterials(_materials.data(), _materials.size());

    // if we update CUDA's copy of the scene, we must also call AnimateScene to supply the transforms
    gpuSceneState.setGeometry(*this);

    DynamicArray<matrix4x4> modelToWorld(_models.size());
    for (size_t i = 0; i < _models.size(); ++i)
        modelToWorld[i] = matrix4x4(_models[i]->getTransform());
    gpuSceneState.animate(modelToWorld.data(), modelToWorld.size());

    gpuSceneState.updateLighting(*this);

    gpuSceneState.fetchUpdatedBVH(_nodes.data());
}

void Raycaster::updateScene(double elapsedTime) {
    (void)elapsedTime;

    if (_sceneDirty) {
        // TODO(anankervis): commitTransforms() or Commmit on the Model class

        buildScene();
        uploadScene();

        _sceneDirty = false;
    }
}

} // namespace hvvr
