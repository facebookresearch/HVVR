/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "camera.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "model.h"
#include "raycaster.h"
#include "raycaster_common.h"
#include "thread_pool.h"
#include "timer.h"

#define DUMP_SCENE_AND_RAYS 0
// If disabled, rays are blocked the same way they are during tracing,
// which should improve coherence but makes visualization more difficult
#define DUMP_IN_SCANLINE_ORDER 1
// dump in binary or text format?
#define DUMP_BINARY 1


namespace hvvr {

#if DUMP_SCENE_AND_RAYS
// Modified version of
// http://segeval.cs.princeton.edu/public/off_format.html for rays:
// RFF
// numRays
// origin.x origin.y origin.z direction.x direction.y direction.z
// origin.x origin.y origin.z direction.x direction.y direction.z
//... numRays like above
static void dumpRaysToRFF(const std::string& filename, const std::vector<SimpleRay>& rays) {
    // Strip nan rays
    std::vector<SimpleRay> culledRays;
    for (const auto& r : rays) {
        if (r.direction.x == r.direction.x) {
            culledRays.push_back(r);
        }
    }

    assert(rays.size() <= INT_MAX);

#if DUMP_BINARY
    FILE* file = fopen((filename + "b").c_str(), "wb");
    char header[4] = {'R', 'F', 'F', 'b'};
    fwrite(header, sizeof(header), 1, file);

    int rayCount = int(culledRays.size());
    fwrite(&rayCount, sizeof(rayCount), 1, file);
    fwrite(rays.data(), sizeof(SimpleRay) * rayCount, 1, file);
    fclose(file);
#else
    FILE* file = fopen(filename.c_str(), "w");
    fprintf(file, "RFF\n");
    fprintf(file, "%d\n", (int)culledRays.size());
    for (auto r : culledRays) {
        fprintf(file, "%.9g %.9g %.9g %.9g %.9g %.9g\n", r.origin.x, r.origin.y, r.origin.z, r.direction.x,
                r.direction.y, r.direction.z);
    }
    fclose(file);
#endif
}

// http://segeval.cs.princeton.edu/public/off_format.html
// OFF
// numVertices numFaces numEdges
// x y z
// x y z
//... numVertices like above
// NVertices v1 v2 v3 ... vN
// MVertices v1 v2 v3 ... vM
//... numFaces like above
static void dumpSceneToOFF(const std::string& filename, const std::vector<std::unique_ptr<Model>>& models) {
    std::vector<vector3> vertices;
    std::vector<uint3> tris;
    for (uint32_t modelIndex = 0; modelIndex < uint32_t(models.size()); modelIndex++) {
        const Model& model = *(models[modelIndex]);
        const MeshData& meshData = model.getMesh();
        for (const auto& t : meshData.triShade) {
            uint32_t curVertexCount = uint32_t(vertices.size());
            tris.push_back(
                {t.indices[0] + curVertexCount, t.indices[1] + curVertexCount, t.indices[2] + curVertexCount});
        }
        for (const auto& v : meshData.verts) {
            vector4 p = model.getTransform() * vector4(v.pos, 1.0);
            vertices.push_back(vector3(p));
        }
    }

    assert(vertices.size() <= INT_MAX);
    assert(tris.size() <= INT_MAX);

#if DUMP_BINARY
    FILE* file = fopen((filename + "b").c_str(), "wb");
    char header[4] = {'O', 'F', 'F', 'b'};
    fwrite(header, sizeof(header), 1, file);

    int vertexCount = int(vertices.size());
    int triCount = int(tris.size());
    int edgeCount = 0;
    fwrite(&vertexCount, sizeof(vertexCount), 1, file);
    fwrite(&triCount, sizeof(triCount), 1, file);
    fwrite(&edgeCount, sizeof(edgeCount), 1, file);

    fwrite(vertices.data(), sizeof(vector3) * vertexCount, 1, file);
    // note - assuming 3 vertices per polygon (triangles)
    fwrite(tris.data(), sizeof(uint3) * triCount, 1, file);
    fclose(file);
#else
    FILE* file = fopen(filename.c_str(), "w");
    fprintf(file, "OFF\n");
    fprintf(file, "%d %d %d\n", (int)vertices.size(), (int)tris.size(), 0);
    for (auto v : vertices) {
        fprintf(file, "%f %f %f\n", v[0], v[1], v[2]);
    }
    for (auto t : tris) {
        fprintf(file, "3 %d %d %d\n", (int)t.x, (int)t.y, (int)t.z);
    }
    fclose(file);
#endif
}

static void dumpBVH(const std::string& filename, const std::vector<std::unique_ptr<Model>>& models) {
    struct DumpedBVHNode {
        uint32_t leafMask;
        union {
            struct Children {
                uint32_t pad0;
                uint32_t offset[4];
            } children;
            struct Leaves {
                // leaf0 = [triIndices[0], triIndices[1]), leaf1 = [triIndices[1], triIndices[2]), ...
                uint32_t triIndices[5];
            } leaves;
        };
    };

    std::vector<DumpedBVHNode> nodes;
    for (uint32_t modelIndex = 0; modelIndex < uint32_t(models.size()); modelIndex++) {
        const Model& model = *(models[modelIndex]);
        const MeshData& meshData = model.getMesh();
        for (const auto& srcNode : meshData.nodes) {
            DumpedBVHNode node;
            memcpy(&node, &srcNode, sizeof(DumpedBVHNode));
            nodes.push_back(node);
        }
    }

    assert(nodes.size() <= INT_MAX);

#if DUMP_BINARY
    FILE* file = fopen((filename + "b").c_str(), "wb");
    char header[4] = {'B', 'F', 'F', 'b'};
    fwrite(header, sizeof(header), 1, file);

    int nodeCount = int(nodes.size());
    fwrite(&nodeCount, sizeof(nodeCount), 1, file);

    fwrite(nodes.data(), sizeof(DumpedBVHNode) * nodeCount, 1, file);
    fclose(file);
#else
    FILE* file = fopen(filename.c_str(), "w");
    fprintf(file, "BFF\n");
    fprintf(file, "%d\n", (int)nodes.size());
    for (const auto& node : nodes) {
        fprintf(file, "%u %u %u %u %u %u\n",
            node.leafMask,
            node.leaves.triIndices[0],
            node.leaves.triIndices[1],
            node.leaves.triIndices[2],
            node.leaves.triIndices[3],
            node.leaves.triIndices[4]);
    }
    fclose(file);
#endif
}

static void dumpSceneAndRays(GPUCamera* gpuCamera,
                             const matrix4x4& cameraToWorld,
                             const std::vector<std::unique_ptr<Model>>& models) {
    static bool dumped = false;
    if (!dumped) {
        Timer timer;

        dumpBVH("bvh_dump.bff", models);
        dumpSceneToOFF("scene_dump.off", models);

        std::vector<SimpleRay> rays;
        gpuCamera->dumpRays(rays, DUMP_IN_SCANLINE_ORDER == 1, cameraToWorld);
        dumpRaysToRFF("ray_dump.rff", rays);

        double dumpTime = timer.get();
        printf("dump time %f\n", dumpTime);

        dumped = true;
    }
}
#endif // DUMP_SCENE_AND_RAYS

void Raycaster::render(double elapsedTime) {
    updateScene(elapsedTime);
    if (_nodes.size() == 0)
        return; // no scene geometry is loaded

    switch (_spec.mode) {
        case RayCasterGPUMode::GPU_INTERSECT_AND_RECONSTRUCT_DEFERRED_MSAA_RESOLVE:
            renderGPUIntersectAndReconstructDeferredMSAAResolve();
            break;
        case RayCasterGPUMode::GPU_FOVEATED_POLAR_SPACE_CUDA_RECONSTRUCT:
            renderFoveatedPolarSpaceCudaReconstruct();
            break;
        default:
            assert(false);
    }
}

static inline void debugPrintTileCost(const BeamBatch& samples, size_t blockCount) {
    double maxCost = 0.0f;
    double meanCost = 0.0f;

    auto frustumCost = [](Frustum f) {
        vector3* dirs = f.pointDir;
        double a0 = (double)length(cross(dirs[0] - dirs[1], dirs[0] - f.pointDir[3]));
        double a1 = (double)length(cross(dirs[2] - dirs[1], dirs[2] - f.pointDir[3]));
        return 0.5 * (a0 + a1);
    };

    double maxBlockCost = 0.0f;
    double meanBlockCost = 0.0f;

    for (size_t i = 0; i < blockCount; ++i) {
        for (size_t j = 0; j < TILES_PER_BLOCK; ++j) {
            const auto& f = samples.tileFrusta3D[i * TILES_PER_BLOCK + j];
            double cost = frustumCost(f);
            maxCost = std::max(cost, maxCost);
            meanCost += cost;
        }
        const auto& f = samples.blockFrusta3D[i];
        double cost = frustumCost(f);
        maxBlockCost = std::max(cost, maxBlockCost);
        meanBlockCost += cost;
    }
    meanCost /= (double)(blockCount * TILES_PER_BLOCK);
    meanBlockCost /= (double)blockCount;
    printf("Mean Tile Cost: %f\nMax Tile Cost: %f\n", meanCost, maxCost);
    printf("Mean Block Cost: %f\nMax Block Cost: %f\n", meanBlockCost, maxBlockCost);
}

void Raycaster::setupAllRenderTargets() {
    // Do DX11/CUDA interop setup if necessary
    for (auto& camera : _cameras) {
        camera->setupRenderTarget(*_gpuContext);
    }
}

void Raycaster::blitAllRenderTargets() {
    _gpuContext->interopMapResources();
    // Copy the results to the camera's DX texture
    for (auto& camera : _cameras) {
        if (!camera->getEnabled())
            continue;
        camera->extractImage();
    }
    _gpuContext->interopUnmapResources();
}

static bool planeCullsFrustum(const Plane plane, const SimpleRayFrustum& frustum) {
    bool allout = true;
    for (int i = 0; i < 4; ++i) {
        allout = allout && dot(plane.normal, frustum.origins[i]) > plane.dist;
        // Extend rays far-out
        allout = allout && dot(plane.normal, frustum.origins[i] + frustum.directions[i] * 10000.0f) > plane.dist;
    }
    return allout;
}


void transformHierarchyCameraToWorld(const Frustum* tilesSrc,
                                     const Frustum* blocksSrc,
                                     Frustum* tilesDst,
                                     Frustum* blocksDst,
                                     const matrix4x4 cameraToWorld,
                                     uint32_t blockCount,
                                     Camera_StreamedData* streamed,
                                     Plane cullPlanes[4],
                                     ThreadPool& threadPool) {
    SimpleRayFrustum* simpleTileFrusta = streamed->tileFrusta3D.dataHost();

    auto blockTransformTask = [&](uint32_t startBlock, uint32_t endBlock) -> void {
        assert((_mm_getcsr() & 0x8040) == 0x8040); // make sure denormals are being treated as zero
        for (uint32_t blockIndex = startBlock; blockIndex < endBlock; blockIndex++) {
            blocksDst[blockIndex] = frustumTransform(blocksSrc[blockIndex], cameraToWorld);

            uint32_t startTile = blockIndex * TILES_PER_BLOCK;
            uint32_t endTile = startTile + TILES_PER_BLOCK;
            for (uint32_t tileIndex = startTile; tileIndex < endTile; tileIndex++) {
                Frustum& tileDst = tilesDst[tileIndex];
                tileDst = frustumTransform(tilesSrc[tileIndex], cameraToWorld);
                SimpleRayFrustum& simpleFrustum = simpleTileFrusta[tileIndex];
                for (int n = 0; n < Frustum::pointCount; n++) {
                    simpleFrustum.origins[n] = tileDst.pointOrigin[n];
                    simpleFrustum.directions[n] = tileDst.pointDir[n];
                }
            }
        }
    };

    auto blockTransformAndCullTask = [&](uint32_t startBlock, uint32_t endBlock) -> void {
        assert((_mm_getcsr() & 0x8040) == 0x8040); // make sure denormals are being treated as zero
        for (uint32_t blockIndex = startBlock; blockIndex < endBlock; blockIndex++) {
            blocksDst[blockIndex] = frustumTransform(blocksSrc[blockIndex], cameraToWorld);

            uint32_t startTile = blockIndex * TILES_PER_BLOCK;
            uint32_t endTile = startTile + TILES_PER_BLOCK;
            for (uint32_t tileIndex = startTile; tileIndex < endTile; tileIndex++) {
                Frustum& tileDst = tilesDst[tileIndex];
                tileDst = frustumTransform(tilesSrc[tileIndex], cameraToWorld);
                SimpleRayFrustum& simpleFrustum = simpleTileFrusta[tileIndex];
                for (int n = 0; n < Frustum::pointCount; n++) {
                    simpleFrustum.origins[n] = tileDst.pointOrigin[n];
                    simpleFrustum.directions[n] = tileDst.pointDir[n];
                }
                bool culled = false;
                for (int i = 0; i < 4; ++i) {
                    culled = culled || planeCullsFrustum(cullPlanes[i], simpleFrustum);
                }
                if (culled) {
                    for (int i = 0; i < 4; ++i) {
                        // Signal degenerate frustum
                        simpleFrustum.origins[i] = vector3(INFINITY, INFINITY, INFINITY);
                        simpleFrustum.directions[i] = vector3(0, 0, 0);
                    }
                    tileDst = Frustum(simpleFrustum.origins, simpleFrustum.directions);
                }
            }
        }
    };


    enum { maxTasks = 4096 };
    enum { blocksPerThread = 16 };
    uint32_t numTasks = (blockCount + blocksPerThread - 1) / blocksPerThread;
    assert(numTasks <= maxTasks);
    numTasks = min<uint32_t>(maxTasks, numTasks);


    bool mustCull = !isinf(cullPlanes[0].dist);
    std::future<void> taskResults[maxTasks];
    for (uint32_t i = 0; i < numTasks; ++i) {
        uint32_t startBlock = min(blockCount, i * blocksPerThread);
        uint32_t endBlock = min(blockCount, (i + 1) * blocksPerThread);
        if (mustCull) {
            taskResults[i] = threadPool.addTask(blockTransformAndCullTask, startBlock, endBlock);
        } else {
            taskResults[i] = threadPool.addTask(blockTransformTask, startBlock, endBlock);
        }
    }
    for (uint32_t i = 0; i < numTasks; ++i) {
        taskResults[i].get();
    }
};

void Raycaster::renderCameraGPUIntersectAndReconstructDeferredMSAAResolve(std::unique_ptr<hvvr::Camera>& camera) {
    if (!camera->getEnabled())
        return;

    const BeamBatch& batch = camera->getSampleData().samples;

    const matrix4x4& cameraToWorld = camera->getCameraToWorld();
    Plane p = Plane::createDegenerate();
    Plane cullPlanes[4] = {p, p, p, p};

    intersectAndResolveBeamBatch(camera, _gpuContext->sceneState, batch, cameraToWorld, cullPlanes);

    camera->_gpuCamera->remap();
}

void Raycaster::renderGPUIntersectAndReconstructDeferredMSAAResolve() {
    if (_spec.outputTo3DApi == true) {
        setupAllRenderTargets();
    }

    // Render and reconstruct from each camera
    for (auto& camera : _cameras) {
        renderCameraGPUIntersectAndReconstructDeferredMSAAResolve(camera);
    }

    if (_spec.outputTo3DApi == true) {
        blitAllRenderTargets();
    }
}

static void cullRectToCullPlanes(const FloatRect cullRect,
                                 Plane cullPlanes[4],
                                 matrix3x3 sampleToCamera,
                                 matrix3x3 cameraToEye,
                                 matrix4x4 eyeToWorld) {
    auto U = cullRect.upper;
    auto L = cullRect.lower;

    vector2 sampleDirs[4] = {{U.x, U.y}, {L.x, U.y}, {L.x, L.y}, {U.x, L.y}};

    const float EPSILON = -0.01f;
    for (int i = 0; i < 4; ++i) {
        vector3 dir0 = sampleToCamera * vector3(sampleDirs[i], 1.0f);
        vector3 dir1 = sampleToCamera * vector3(sampleDirs[(i + 1) % 4], 1.0f);
        Plane eyeSpacePlane;
        eyeSpacePlane.normal = cameraToEye * normalize(cross(dir1, dir0));
        eyeSpacePlane.dist = EPSILON;
        cullPlanes[i] = eyeToWorld * eyeSpacePlane;
    }
}

void Raycaster::intersectAndResolveBeamBatch(std::unique_ptr<Camera>& camera,
                                             GPUSceneState& scene,
                                             const BeamBatch& batch,
                                             const matrix4x4& batchToWorld,
                                             Plane cullPlanes[4]) {
    GPUCamera* gpuCamera = camera->_gpuCamera;
    uint32_t tileCount = uint32_t(batch.tileFrusta3D.size());

    // begin filling data for the GPU
    Camera_StreamedData* streamed = gpuCamera->streamedDataLock(tileCount);
    {
        camera->advanceJitter();

        Camera::CPUHierarchy& cHier = camera->_cpuHierarchy;

        transformHierarchyCameraToWorld(batch.tileFrusta3D.data(), batch.blockFrusta3D.data(), cHier._tileFrusta.data(),
                                        cHier._blockFrusta.data(), batchToWorld, uint32_t(batch.blockFrusta3D.size()),
                                        streamed, cullPlanes, *_threadPool.get());

        RayHierarchy rayHierachy;
        rayHierachy.blockFrusta = cHier._blockFrusta;
        rayHierachy.tileFrusta = cHier._tileFrusta;

#if DUMP_SCENE_AND_RAYS
        dumpSceneAndRays(gpuCamera, camera->getCameraToWorld(), _models);
#endif

        buildTileTriangleLists(rayHierachy, streamed);

        // end filling data for the GPU
    }
    gpuCamera->streamedDataUnlock();
    gpuCamera->intersectShadeResolve(scene, batchToWorld);
};

// TODO(anankervis): merge into a single render path
void Raycaster::renderFoveatedPolarSpaceCudaReconstruct() {
    if (_spec.outputTo3DApi == true) {
        setupAllRenderTargets();
    }
    polarSpaceFoveatedSetup(this);
    {
        for (auto& camera : _cameras) {
            if (!camera->getEnabled())
                continue;

            const vector3& eyeDirection = camera->getEyeDir();
            matrix3x3 eyeToCamera = matrix3x3::rotationFromZAxis(-eyeDirection);
            matrix4x4 cameraToWorld = camera->getCameraToWorld();
            matrix4x4 eyeToWorld = cameraToWorld * matrix4x4(eyeToCamera);

            Plane cullPlanes[4];
            cullRectToCullPlanes(camera->getSampleData().sampleBounds, cullPlanes, camera->getSampleToCamera(),
                                 invert(eyeToCamera), eyeToWorld);
            const BeamBatch& samples = camera->_foveatedSampleData.samples;

            intersectAndResolveBeamBatch(camera, _gpuContext->sceneState, samples, eyeToWorld, cullPlanes);

            camera->_gpuCamera->remapPolarFoveated();

            matrix3x3 sampleToEye = invert(eyeToCamera) * camera->getSampleToCamera();
            matrix4x4 eyeToEyePrevious = camera->_worldToEyePrevious * eyeToWorld;

            // Handles Temporal Filtering if necessary
            camera->_gpuCamera->foveatedPolarToScreenSpace(eyeToEyePrevious, camera->_eyePreviousToSamplePrevious,
                                                           sampleToEye);

            camera->_worldToEyePrevious = invert(eyeToWorld);
            camera->_eyePreviousToSamplePrevious = invert(sampleToEye);
        }
    }
    if (_spec.outputTo3DApi == true) {
        blitAllRenderTargets();
    }
}

} // namespace hvvr
