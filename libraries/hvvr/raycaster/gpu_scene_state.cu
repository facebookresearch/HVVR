/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bvh_node.h"
#include "cuda_util.h"
#include "gpu_scene_state.h"
#include "graphics_types.h"
#include "kernel_constants.h"
#include "memory_helpers.h"
#include "model.h"
#include "raycaster.h"
#include "sort.h"
#include "vector_math.h"
#include "warp_ops.h"
#include "dynamic_array.h"
#include <algorithm>


namespace hvvr {

GPUSceneState::GPUSceneState() : updateEvent(0), stream(0) {
    reset();
}

GPUSceneState::~GPUSceneState() {
    reset();
}

void GPUSceneState::reset() {
    if (updateEvent) {
        cutilSafeCall(cudaEventDestroy(updateEvent));
        updateEvent = nullptr;
    }
    if (stream) {
        cutilSafeCall(cudaStreamDestroy(stream));
        stream = nullptr;
    }

    updateDirty = true;
    updatedBVHAvailable = false;
}

void GPUSceneState::updateLighting(const Raycaster& raycaster) {
    LightingEnvironment& env = lightingEnvironment;
    memset(&env, 0, sizeof(LightingEnvironment));

    for (const auto& light : raycaster._lights) {
        const LightUnion& lightUnion = light->getLightUnion();

        switch (lightUnion.type) {
        case LightType::directional:
            if (env.directionalLightCount < MAX_DIRECTIONAL_LIGHTS) {
                env.directionalLights[env.directionalLightCount++] = lightUnion.directional;
            } else {
                fprintf(stderr, "Too many directional lights for raycaster!");
            }
            break;
        case LightType::point:
            if (env.pointLightCount < MAX_POINT_LIGHTS) {
                env.pointLights[env.pointLightCount++] = lightUnion.point;
            } else {
                fprintf(stderr, "Too many point lights for raycaster!");
            }
            break;
        case LightType::spot:
            if (env.spotLightCount < MAX_SPOT_LIGHTS) {
                env.spotLights[env.spotLightCount++] = lightUnion.spot;
            } else {
                fprintf(stderr, "Too many spot lights for raycaster!");
            }
            break;
        default:
            assert(false);
            break;
        }
    }
}

void GPUSceneState::updateMaterials(SimpleMaterial* _materials, size_t materialCount) {
    materials = GPUBuffer<SimpleMaterial>(_materials, _materials + materialCount);
}

// transform verts, generate precomputed verts
CUDA_KERNEL void SceneTransformVerticesKernel(const ShadingVertex* CUDA_RESTRICT inputVertices,
                                              ShadingVertex* outputVertices,
                                              const uint32_t* CUDA_RESTRICT vertexToTransformMapping,
                                              const matrix4x4* CUDA_RESTRICT meshToWorldTransforms,
                                              uint32_t vertexCount) {
    // let's play some games here to get better memory access coalescing...
    __shared__ union SMem {
        SMem() {}

        ShadingVertex vertices[WARP_SIZE];
        uint32_t chunks[1];
    } sMem;
    uint32_t vertStart = blockIdx.x * WARP_SIZE;
    uint32_t vertStop = min(vertStart + WARP_SIZE, vertexCount);
    uint32_t localVertexCount = vertStop - vertStart;
    uint32_t chunkCount = (sizeof(ShadingVertex) * localVertexCount) / sizeof(uint32_t);
    const uint32_t* CUDA_RESTRICT srcChunks = (const uint32_t* CUDA_RESTRICT)(inputVertices + vertStart);
    for (uint32_t chunkIndex = threadIdx.x; chunkIndex < chunkCount; chunkIndex += WARP_SIZE) {
        sMem.chunks[chunkIndex] = srcChunks[chunkIndex];
    }
    __syncthreads();

    uint32_t localIndex = threadIdx.x;
    uint32_t globalIndex = vertStart + localIndex;
    if (localIndex < localVertexCount) {
        ShadingVertex& vertex = sMem.vertices[localIndex];

        uint32_t transformIndex = vertexToTransformMapping[globalIndex];
        matrix4x4 transform = meshToWorldTransforms[transformIndex];

        // Need to explicitly divide here because the transformation was not necessarily
        // affine. Common reason for this: bone weights didn't add up to 1.
        vector4 finalPosition = transform * vector4(vertex.pos, 1.0);
        vertex.pos = vector3(finalPosition) * (1.0f / finalPosition.w);

        // Assume uniform scale (so we don't need inverse-transpose)
        vertex.normal = vector4h(normalize(matrix3x3(transform) * vector3(vertex.normal)));
    }

    uint32_t* dstChunks = (uint32_t*)(outputVertices + vertStart);
    __syncthreads();
    for (uint32_t chunkIndex = threadIdx.x; chunkIndex < chunkCount; chunkIndex += WARP_SIZE) {
        dstChunks[chunkIndex] = sMem.chunks[chunkIndex];
    }
}

template <bool SinglePassInternalRefit>
CUDA_KERNEL void SceneRefitPreKernel(uint32_t nodeCount, BVHNode* nodes, uint32_t* nodeReady) {
    if (SinglePassInternalRefit) {
        uint32_t globalIndex = blockIdx.x * WARP_SIZE + threadIdx.x;
        if (globalIndex < nodeCount)
            nodeReady[globalIndex] = 0;
    }

    // We want to order our writes for memory coalescing, by writing contiguous 32-byte chunks for each group of 8
    // threads. Also, we don't want to modify the last 32 (out of 128) bytes in each BVH, as we're just trying to
    // reset the bounding boxes, not the child/leaf data.
    static_assert(sizeof(BVHNode) == 128, "SceneRefitPreKernel BVHNode size");
    static_assert(offsetof(BVHNode, boxData) == 96, "SceneRefitPreKernel boxData offset");

    enum { lineSize = 32 };
    enum { threadsPerLine = lineSize / sizeof(uint32_t) };
    enum { linesPerWarp = WARP_SIZE / threadsPerLine };

    uint32_t nodeStart = blockIdx.x * WARP_SIZE;
    uint32_t nodeStop = min(nodeStart + WARP_SIZE, nodeCount);
    uint32_t localNodeCount = nodeStop - nodeStart;
    uint32_t* dstChunks = (uint32_t*)(nodes + nodeStart);

    uint32_t lineCount = (sizeof(BVHNode) * localNodeCount) / lineSize;
    uint32_t modifiedLineCount = lineCount * 3 / 4;
    for (uint32_t line = threadIdx.x / threadsPerLine; line < modifiedLineCount; line += linesPerWarp) {
        uint32_t modifiedLine = line * 4 / 3;
        uint32_t chunkIndex = modifiedLine * threadsPerLine + threadIdx.x % threadsPerLine;

        dstChunks[chunkIndex] = FloatFlip(0xff800000); // negative infinity, in a form for integer comparison
    }
}

CUDA_KERNEL void ScenePrecomputeTrianglesKernel(uint32_t triangleCount,
                                                const ShadingVertex* CUDA_RESTRICT worldSpaceVertices,
                                                const PrecomputedTriangleShade* CUDA_RESTRICT triangles3DShadeIn,
                                                const uint32_t* CUDA_RESTRICT triToNodeChild,
                                                PrecomputedTriangleIntersect* triangles3DIntersectOut,
                                                BVHNode* nodes) {
    // let's play some games here to get better memory access coalescing...
    __shared__ union SMem {
        SMem() {}

        PrecomputedTriangleShade triShade[WARP_SIZE];
        PrecomputedTriangleIntersect triOut[WARP_SIZE];
        uint32_t chunks[1];
    } sMem;
    uint32_t triStart = blockIdx.x * WARP_SIZE;
    uint32_t triStop = min(triStart + WARP_SIZE, triangleCount);
    uint32_t localTriCount = triStop - triStart;
    uint32_t chunkCountIn = (sizeof(PrecomputedTriangleShade) * localTriCount) / sizeof(uint32_t);
    const uint32_t* CUDA_RESTRICT srcChunks = (const uint32_t* CUDA_RESTRICT)(triangles3DShadeIn + triStart);
    for (uint32_t chunkIndex = threadIdx.x; chunkIndex < chunkCountIn; chunkIndex += WARP_SIZE) {
        sMem.chunks[chunkIndex] = srcChunks[chunkIndex];
    }
    __syncthreads();

    uint32_t localIndex = threadIdx.x;
    uint32_t globalIndex = triStart + localIndex;

    uint32_t vertIndices[3];
    if (localIndex < localTriCount) {
        const PrecomputedTriangleShade& tri3DShade = sMem.triShade[localIndex];
        vertIndices[0] = tri3DShade.indices[0];
        vertIndices[1] = tri3DShade.indices[1];
        vertIndices[2] = tri3DShade.indices[2];
    }

    __syncthreads(); // alias sMem
    if (localIndex < localTriCount) {
        PrecomputedTriangleIntersect& triOut = sMem.triOut[localIndex];

        vector3 v0 = worldSpaceVertices[vertIndices[0]].pos;
        vector3 v1 = worldSpaceVertices[vertIndices[1]].pos;
        vector3 v2 = worldSpaceVertices[vertIndices[2]].pos;

        triOut.v0 = v0;
        triOut.edge0 = v1 - v0;
        triOut.edge1 = v2 - v0;

        // update BVH leaf node bounds
        vector3 triMax;
        triMax.x = max(v0.x, max(v1.x, v2.x));
        triMax.y = max(v0.y, max(v1.y, v2.y));
        triMax.z = max(v0.z, max(v1.z, v2.z));

        vector3 triNegMin;
        triNegMin.x = max(-v0.x, max(-v1.x, -v2.x));
        triNegMin.y = max(-v0.y, max(-v1.y, -v2.y));
        triNegMin.z = max(-v0.z, max(-v1.z, -v2.z));

        uint32_t nodeChild = triToNodeChild[globalIndex];
        uint32_t parentIndex = nodeChild >> 2;
        uint32_t childSlot = nodeChild & ((1 << 2) - 1);
        BVHNode& parentNode = nodes[parentIndex];

        atomicMax((uint32_t*)&parentNode.xMax[childSlot], FloatFlipF(triMax.x));
        atomicMax((uint32_t*)&parentNode.xNegMin[childSlot], FloatFlipF(triNegMin.x));
        atomicMax((uint32_t*)&parentNode.yMax[childSlot], FloatFlipF(triMax.y));
        atomicMax((uint32_t*)&parentNode.yNegMin[childSlot], FloatFlipF(triNegMin.y));
        atomicMax((uint32_t*)&parentNode.zMax[childSlot], FloatFlipF(triMax.z));
        atomicMax((uint32_t*)&parentNode.zNegMin[childSlot], FloatFlipF(triNegMin.z));
    }

    uint32_t chunkCountOut = (sizeof(PrecomputedTriangleIntersect) * localTriCount) / sizeof(uint32_t);
    uint32_t* dstChunks = (uint32_t*)(triangles3DIntersectOut + triStart);
    __syncthreads();
    for (uint32_t chunkIndex = threadIdx.x; chunkIndex < chunkCountOut; chunkIndex += WARP_SIZE) {
        dstChunks[chunkIndex] = sMem.chunks[chunkIndex];
    }
}

CUDA_KERNEL void SceneRefitIntFixupKernel(uint32_t nodeCount, BVHNode* nodes) {
    // We want to order our writes for memory coalescing, by writing contiguous 32-byte chunks for each group of 8
    // threads. Also, we don't want to modify the last 32 (out of 128) bytes in each BVH, as we're just trying to
    // reset the bounding boxes, not the child/leaf data.
    static_assert(sizeof(BVHNode) == 128, "SceneRefitIntFixupKernel BVHNode size");
    static_assert(offsetof(BVHNode, boxData) == 96, "SceneRefitIntFixupKernel boxData offset");

    enum { lineSize = 32 };
    enum { threadsPerLine = lineSize / sizeof(uint32_t) };
    enum { linesPerWarp = WARP_SIZE / threadsPerLine };

    uint32_t nodeStart = blockIdx.x * WARP_SIZE;
    uint32_t nodeStop = min(nodeStart + WARP_SIZE, nodeCount);
    uint32_t localNodeCount = nodeStop - nodeStart;
    uint32_t* dstChunks = (uint32_t*)(nodes + nodeStart);

    uint32_t lineCount = (sizeof(BVHNode) * localNodeCount) / lineSize;
    uint32_t modifiedLineCount = lineCount * 3 / 4;
    for (uint32_t line = threadIdx.x / threadsPerLine; line < modifiedLineCount; line += linesPerWarp) {
        uint32_t modifiedLine = line * 4 / 3;
        uint32_t chunkIndex = modifiedLine * threadsPerLine + threadIdx.x % threadsPerLine;

        dstChunks[chunkIndex] = IFloatFlip(dstChunks[chunkIndex]);
    }
}

// each child computes their own bounds, and pushes that into their slot in the parent
CUDA_KERNEL void SceneRefitNodeSinglePassKernel(const SceneRefitTaskNode* CUDA_RESTRICT nodeTasks,
                                                volatile uint32_t* nodeReady,
                                                uint32_t taskCount,
                                                volatile BVHNode* nodes) {
    while (true) {
        // nodeReady[0] corresponds to the root node, which doesn't execute a task, so we can
        // repurpose it as a work item counter
        uint32_t taskGroupIndex;
        if (threadIdx.x == 0) {
            taskGroupIndex = atomicAdd((uint32_t*)nodeReady, 1);
        }
        taskGroupIndex = laneBroadcast(taskGroupIndex, 0);

        uint32_t taskIndex = taskGroupIndex * WARP_SIZE + threadIdx.x;
        if (taskIndex >= taskCount)
            break; // we've run out of tasks, this thread can die

        const SceneRefitTaskNode& nodeTask = nodeTasks[taskIndex];

        uint32_t childIndex = nodeTask.childIndex;
        const volatile BVHNode& childNode = nodes[childIndex];

        uint32_t parentIndex = nodeTask.nodeChild >> 2;
        uint32_t childSlot = nodeTask.nodeChild & ((1 << 2) - 1);
        volatile BVHNode& parentNode = nodes[parentIndex];

        // leaf nodes are guaranteed to be ready by this point (they don't contribute to nodeReady's mask)
        uint32_t readyMask = ~childNode.boxData.leafMask & 0xf;
        // The CUDA compiler seems to have a bug where, depending on how this loop is structured, it will
        // loop forever waiting for ALL threads to become ready, then execute the inside of the conditional
        // once.
        bool run = true;
        while (run) {
            if (nodeReady[childIndex] == readyMask) {
                const volatile float* boundsPtr = childNode.xMax;
                float childBounds[6];
                for (int n = 0; n < 6; n++) {
                    childBounds[n] = max(boundsPtr[n * 4 + 0],
                                         max(boundsPtr[n * 4 + 1], max(boundsPtr[n * 4 + 2], boundsPtr[n * 4 + 3])));
                }

                parentNode.xMax[childSlot] = childBounds[0];
                parentNode.xNegMin[childSlot] = childBounds[1];
                parentNode.yMax[childSlot] = childBounds[2];
                parentNode.yNegMin[childSlot] = childBounds[3];
                parentNode.zMax[childSlot] = childBounds[4];
                parentNode.zNegMin[childSlot] = childBounds[5];

                // ensure preceding writes come before the atomicOr
                __threadfence();

                atomicOr((uint32_t*)nodeReady + parentIndex, 1 << childSlot);

                run = false;
            }
        }
    }
}

// each child computes their own bounds, and pushes that into their slot in the parent
CUDA_KERNEL void SceneRefitNodeMultiPassKernel(const SceneRefitTaskNode* CUDA_RESTRICT nodeTasks,
                                               uint32_t taskCount,
                                               BVHNode* nodes) {
    uint32_t taskWarpBase = blockIdx.x * WARP_SIZE;
    uint32_t taskIndex = taskWarpBase + threadIdx.x;
    if (taskIndex >= taskCount) {
        taskIndex = taskWarpBase; // keep extra threads alive as helpers
    }

    const SceneRefitTaskNode& nodeTask = nodeTasks[taskIndex];

    enum { boundsComponents = 24 };
    static_assert(boundsComponents < WARP_SIZE, "SceneRefitNodeMultiPassKernel boundsComponents < WARP_SIZE");
    __shared__ float childData[WARP_SIZE * boundsComponents];
    for (int lane = 0; lane < WARP_SIZE; lane++) {
        uint32_t childIndex = laneBroadcast(nodeTask.childIndex, lane);
        const float* CUDA_RESTRICT boundsPtr = (const float* CUDA_RESTRICT)nodes[childIndex].xMax;

        if (threadIdx.x < boundsComponents) {
            childData[lane * boundsComponents + threadIdx.x] = boundsPtr[threadIdx.x];
        }
    }
    __syncthreads();

    float childBounds[6];
    for (int n = 0; n < 6; n++) {
        childBounds[n] = max(
            childData[threadIdx.x * boundsComponents + n * 4 + 0],
            max(childData[threadIdx.x * boundsComponents + n * 4 + 1],
                max(childData[threadIdx.x * boundsComponents + n * 4 + 2], childData[threadIdx.x * 24 + n * 4 + 3])));
    }

    uint32_t parentIndex = nodeTask.nodeChild >> 2;
    uint32_t childSlot = nodeTask.nodeChild & ((1 << 2) - 1);
    BVHNode& parentNode = nodes[parentIndex];

    parentNode.xMax[childSlot] = childBounds[0];
    parentNode.xNegMin[childSlot] = childBounds[1];
    parentNode.yMax[childSlot] = childBounds[2];
    parentNode.yNegMin[childSlot] = childBounds[3];
    parentNode.zMax[childSlot] = childBounds[4];
    parentNode.zNegMin[childSlot] = childBounds[5];
}

void GPUSceneState::update() {
    if (!updateDirty)
        return;

    enum { singlePassCutoff = 1024 * 64 };
    uint32_t refitTaskCount = uint32_t(refitTasksNode.size());
    bool singlePassInternalRefit = (refitTaskCount <= singlePassCutoff);

    // transform vertices
    {
        uint32_t vertexCount = uint32_t(untransformedVertices.size());
        KernelDim dim = KernelDim(vertexCount, WARP_SIZE);
        SceneTransformVerticesKernel<<<dim.grid, dim.block, 0, stream>>>(
            untransformedVertices, worldSpaceVertices, vertexToTransformMapping, _modelToWorld, vertexCount);
    }

    // initialize BVH bounds
    uint32_t nodeCount = uint32_t(nodes.size());
    {
        KernelDim dim = KernelDim(nodeCount, WARP_SIZE);
        if (singlePassInternalRefit) {
            SceneRefitPreKernel<true><<<dim.grid, dim.block, 0, stream>>>(nodeCount, nodes, nodeReady);
        } else {
            SceneRefitPreKernel<false><<<dim.grid, dim.block, 0, stream>>>(nodeCount, nodes, nodeReady);
        }
    }

    // precompute tris for intersection, and update BVH leaves
    {
        size_t triangleCount = trianglesIntersect.size();
        KernelDim dim = KernelDim(triangleCount, WARP_SIZE);
        ScenePrecomputeTrianglesKernel<<<dim.grid, dim.block, 0, stream>>>(
            uint32_t(triangleCount), worldSpaceVertices, trianglesShade, triToNodeChild, trianglesIntersect, nodes);
    }

    // fixup after atomic ops on integer representation of floats
    {
        KernelDim dim = KernelDim(nodeCount, WARP_SIZE);
        SceneRefitIntFixupKernel<<<dim.grid, dim.block, 0, stream>>>(nodeCount, nodes);
    }

    // update BVH internal nodes
    {
        if (singlePassInternalRefit) {
            // single pass to minimize kernel launch overhead
            uint32_t nodeTaskCount = refitTaskCount;
            uint32_t warpCount = 30 * 4 * 6; // enough to hit occupancy of 6 on P6000
            KernelDim dim(warpCount * WARP_SIZE, WARP_SIZE);
            SceneRefitNodeSinglePassKernel<<<dim.grid, dim.block, 0, stream>>>(refitTasksNode, nodeReady, nodeTaskCount,
                                                                               nodes);
        } else {
            // one pass for each level of BVH depth, starting from the bottom
            uint32_t groupCount = uint32_t(refitNodeGroupBoundaries.size());
            for (uint32_t group = 0; group < groupCount; group++) {
                uint32_t groupEndIndex = refitNodeGroupBoundaries[group] + 1;
                uint32_t groupStartIndex = 0;
                if (group > 0) {
                    groupStartIndex = refitNodeGroupBoundaries[group - 1] + 1;
                    // we shouldn't see the same boundary twice - that implies an empty group
                    assert(groupEndIndex != groupStartIndex);
                }
                uint32_t nodeTaskCount = groupEndIndex - groupStartIndex;

                KernelDim dimNode = KernelDim(nodeTaskCount, WARP_SIZE);
                SceneRefitNodeMultiPassKernel<<<dimNode.grid, dimNode.block, 0, stream>>>(
                    refitTasksNode + groupStartIndex, nodeTaskCount, nodes);
            }
        }

        updatedBVHAvailable = true;
    }

    cutilSafeCall(cudaEventRecord(updateEvent, stream));
    updateDirty = false;
}

// copy the GPU's BVH back to the CPU, for CPU traversal
bool GPUSceneState::fetchUpdatedBVH(BVHNode* dstNodes) {
    if (updateDirty) {
        update();
    }

    if (!updatedBVHAvailable)
        return false;

    cutilSafeCall(cudaEventSynchronize(updateEvent));
    nodes.readback(dstNodes);

    updatedBVHAvailable = false;
    return true;
}

void GPUSceneState::setGeometry(const Raycaster& raycaster) {
    uint32_t globalVertexCount = raycaster._vertexCount;
    uint32_t globalTriCount = uint32_t(raycaster._trianglesShade.size());
    uint32_t globalNodeCount = uint32_t(raycaster._nodes.size());
    uint32_t modelCount = uint32_t(raycaster._models.size());

    if (stream == 0) {
        cutilSafeCall(cudaStreamCreate(&stream));
        cutilSafeCall(cudaEventCreateWithFlags(&updateEvent, cudaEventDisableTiming));
    }

    // convert data into GPU layout
    auto verticesCPU = DynamicArray<ShadingVertex>(globalVertexCount);
    auto verticesToTransformRemappingCPU = DynamicArray<uint32_t>(globalVertexCount);
    {
        uint32_t vertexOffset = 0;
        for (uint32_t modelIndex = 0; modelIndex < modelCount; modelIndex++) {
            const Model& model = *(raycaster._models[modelIndex]);
            const MeshData& meshData = model.getMesh();

            uint32_t vertexCount = uint32_t(meshData.verts.size());
            for (uint32_t n = 0; n < vertexCount; n++) {
                verticesToTransformRemappingCPU[vertexOffset + n] = modelIndex;
                verticesCPU[vertexOffset + n] = meshData.verts[n];
            }
            vertexOffset += vertexCount;
        }
    }

    // copy to GPU
    trianglesIntersect = GPUBuffer<PrecomputedTriangleIntersect>(globalTriCount);
    trianglesShade = makeGPUBuffer(raycaster._trianglesShade);
    untransformedVertices = GPUBuffer<ShadingVertex>(verticesCPU.begin(), verticesCPU.end());
    _modelToWorld = GPUBuffer<matrix4x4>(modelCount);
    worldSpaceVertices = GPUBuffer<ShadingVertex>(verticesCPU.size());
    vertexToTransformMapping = makeGPUBuffer(verticesToTransformRemappingCPU);

    // measure BVH node depths
    uint32_t maxInternalDepth = 0;
    std::vector<uint32_t> nodeDepths(globalNodeCount);
    nodeDepths[0] = 0; // root node depth
    for (auto node = raycaster._nodes.begin(); node != raycaster._nodes.end(); ++node) {
        uint32_t nodeIndex = uint32_t(node - raycaster._nodes.data());
        uint32_t depth = nodeDepths[nodeIndex];

        for (int k = 3; k >= 0; k--) {
            if (!_bittest(&node->boxData.leafMask, k)) {
                // child is an internal node
                uint32_t childIndex = nodeIndex + node->boxData.children.offset[k];

                nodeDepths[childIndex] = depth + 1;
                maxInternalDepth = max(depth + 1, maxInternalDepth);
            }
        }
    }

    // generate BVH refit tasks
    std::vector<uint32_t> triToNodeChildCPU;
    std::vector<SceneRefitTaskNode> refitTasksNodeCPU;
    triToNodeChildCPU.resize(globalTriCount);
    refitTasksNodeCPU.reserve(globalNodeCount);
    for (auto node = raycaster._nodes.end() - 1; node >= raycaster._nodes.begin(); --node) {
        uint32_t parentIndex = uint32_t(node - raycaster._nodes.data());

        for (int k = 3; k >= 0; k--) {
            uint32_t childSlot = k;
            assert(parentIndex < (1 << 30));
            uint32_t nodeChild = (parentIndex << 2) | childSlot;

            if (!_bittest(&node->boxData.leafMask, k)) {
                // child is an internal node
                uint32_t childIndex = parentIndex + node->boxData.children.offset[k];

                SceneRefitTaskNode taskNode = {};
                taskNode.childIndex = childIndex;
                taskNode.nodeChild = nodeChild;

                refitTasksNodeCPU.push_back(taskNode);
            } else {
                // child is a leaf node
                uint32_t triStart = node->boxData.leaf.triIndex[k];
                uint32_t triEnd = node->boxData.leaf.triIndex[k + 1];
                for (uint32_t triIndex = triStart; triIndex < triEnd; triIndex++) {
                    triToNodeChildCPU[triIndex] = nodeChild;
                }
            }
        }
    }

    // sort node tasks by highest depth first, to minimize dependencies between nearby work items
    // for nodes with the same depth, sort by node index to group siblings together
    auto nodeRefitSort = [&nodeDepths](const SceneRefitTaskNode& a, const SceneRefitTaskNode& b) -> bool {
        uint32_t depthA = nodeDepths[a.nodeChild >> 2];
        uint32_t depthB = nodeDepths[b.nodeChild >> 2];

        if (depthA > depthB)
            return true;
        if (depthA < depthB)
            return false;
        return a.childIndex < b.childIndex;
    };
    std::sort(refitTasksNodeCPU.begin(), refitTasksNodeCPU.end(), nodeRefitSort);

    // place internal node refit tasks into groups that can run concurrently (all nodes of the same depth)
    // the root node doesn't have an entry in the task list - children populate the parent node
    refitNodeGroupBoundaries = DynamicArray<uint32_t>(maxInternalDepth);
    uint32_t currentDepth = nodeDepths[refitTasksNodeCPU[0].nodeChild >> 2];
    assert(currentDepth < maxInternalDepth);
    uint32_t currentGroup = 0;
    refitNodeGroupBoundaries[0] = 0;
    for (size_t n = 0; n < refitTasksNodeCPU.size(); n++) {
        uint32_t depth = nodeDepths[refitTasksNodeCPU[n].nodeChild >> 2];
        if (depth != currentDepth) {
            currentDepth = depth;
            currentGroup++;
            assert(currentDepth < maxInternalDepth);
            assert(currentGroup < maxInternalDepth);
        }

        refitNodeGroupBoundaries[currentGroup] = uint32_t(n);
    }

    // copy to GPU
    nodes = makeGPUBuffer(raycaster._nodes);
    triToNodeChild = makeGPUBuffer(triToNodeChildCPU);
    refitTasksNode = makeGPUBuffer(refitTasksNodeCPU);

    nodeReady.resizeDestructive(globalNodeCount);

    updateDirty = true;
}

void GPUSceneState::animate(const matrix4x4* modelToWorld, size_t modelToWorldCount) {
    assert(modelToWorldCount == _modelToWorld.size());
    _modelToWorld.uploadAsync(modelToWorld, 0, modelToWorldCount, stream);

    updateDirty = true;
}

} // namespace hvvr
