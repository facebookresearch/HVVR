/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "camera.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "model.h"
#include "raycaster.h"
#include "raycaster_common.h"
#include "timer.h"
#include "thread_pool.h"

#define DUMP_SCENE_AND_RAYS 0
 // If disabled, rays are blocked the same way they are during tracing, 
 // which should improve coherence but makes visualization more difficult
#define DUMP_IN_SCANLINE_ORDER 0
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
#endif // DUMP_SCENE_AND_RAYS

void Raycaster::render(double elapsedTime) {
    updateScene(elapsedTime);
    if (_nodes.size() == 0)
        return; // no scene geometry is loaded

    switch (_spec.mode) {
        case RayCasterSpecification::GPUMode::GPU_INTERSECT_AND_RECONSTRUCT_DEFERRED_MSAA_RESOLVE:
            renderGPUIntersectAndReconstructDeferredMSAAResolve();
            break;
        case RayCasterSpecification::GPUMode::GPU_FOVEATED_POLAR_SPACE_CUDA_RECONSTRUCT:
            renderFoveatedPolarSpaceCudaReconstruct();
            break;
        default:
            assert(false);
    }
}

static inline void debugPrintTileCost(const SampleHierarchy& samples, size_t blockCount) {
    double maxCost = 0.0f;
    double meanCost = 0.0f;

    double maxBlockCost = 0.0f;
    double meanBlockCost = 0.0f;

    for (size_t i = 0; i < blockCount; ++i) {
        for (size_t j = 0; j < TILES_PER_BLOCK; ++j) {
            const auto& f = samples.tileFrusta2D[i * TILES_PER_BLOCK + j];
            double cost = (f.xMax() - f.xMin()) + (f.yMax() - f.yMin());
            maxCost = std::max(cost, maxCost);
            meanCost += cost;
        }
        const auto& f = samples.blockFrusta2D[i];
        double cost = (f.xMax() - f.xMin()) + (f.yMax() - f.yMin());
        maxBlockCost = std::max(cost, maxBlockCost);
        meanBlockCost += cost;
    }
    meanCost /= (double)(blockCount * TILES_PER_BLOCK);
    meanBlockCost /= (double)blockCount;
    printf("Mean Tile Cost: %f\nMax Tile Cost: %f\n", meanCost, maxCost);
    printf("Mean Block Cost: %f\nMax Block Cost: %f\n", meanBlockCost, maxBlockCost);
}

inline static FloatRect expandRect(const FloatRect& rect, const float fractionToExpand) {
    const vector2 extent = {rect.upper.x - rect.lower.x, rect.upper.y - rect.lower.y};
    const vector2 newLower = {rect.lower.x - (extent.x * fractionToExpand),
                              rect.lower.y - (extent.y * fractionToExpand)};
    const vector2 newUpper = {rect.upper.x + (extent.x * fractionToExpand),
                              rect.upper.y + (extent.y * fractionToExpand)};
    return {newLower, newUpper};
}

void Raycaster::renderGPUIntersectAndReconstructDeferredMSAAResolve() {
#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    // Do DX11/CUDA interop setup if necessary
    for (auto& camera : _cameras) {
        if (!camera->getEnabled())
            continue;
        GPUCamera* gpuCamera = camera->_gpuCamera;

        if (camera->_renderTarget.isHardwareRenderTarget() && camera->_newHardwareTarget) {
            gpuCamera->bindTexture(*_gpuContext, camera->_renderTarget);
            camera->_newHardwareTarget = false;
        }
    }
    _gpuContext->interopMapResources();
#endif
    {
        // Render and reconstruct from each camera
        for (auto& camera : _cameras) {
            if (!camera->getEnabled())
                continue;

            const SampleHierarchy& samples = camera->getSampleData().samples;
            uint32_t tileCount = uint32_t(samples.tileFrusta3D.size());

            GPUCamera* gpuCamera = camera->_gpuCamera;
            // begin filling data for the GPU
            Camera_StreamedData* streamed = gpuCamera->streamedDataLock(tileCount);

            vector2 jitter = camera->_frameJitters[camera->_frameCount % camera->_frameJitters.size()];
            gpuCamera->setCameraJitter(vector2(jitter.x * 0.5f + 0.5f, jitter.y * 0.5f + 0.5f));
            ++camera->_frameCount;

            matrix4x4 cameraToWorld = camera->getCameraToWorld();
            matrix4x4 cameraToWorldInvTrans = transpose(invert(cameraToWorld));

            // transform from camera to world space
            {
                uint32_t blockCount = uint32_t(samples.blockFrusta3D.size());
                const RayPacketFrustum3D* blocksSrc = samples.blockFrusta3D.data();
                RayPacketFrustum3D* blocksDst = camera->_blockFrustaTransformed.data();

                const RayPacketFrustum3D* tilesSrc = samples.tileFrusta3D.data();
                RayPacketFrustum3D* tilesDst = camera->_tileFrustaTransformed.data();

                SimpleRayFrustum* simpleTileFrusta = streamed->tileFrusta3D.dataHost();

                auto blockTransformTask = [&](uint32_t startBlock, uint32_t endBlock) -> void {
                    assert((_mm_getcsr() & 0x8040) == 0x8040); // make sure denormals are being treated as zero

                    for (uint32_t blockIndex = startBlock; blockIndex < endBlock; blockIndex++) {
                        blocksDst[blockIndex] = blocksSrc[blockIndex].transform(cameraToWorld, cameraToWorldInvTrans);

                        uint32_t startTile = blockIndex * TILES_PER_BLOCK;
                        uint32_t endTile = startTile + TILES_PER_BLOCK;
                        for (uint32_t tileIndex = startTile; tileIndex < endTile; tileIndex++) {
                            RayPacketFrustum3D& tileDst = tilesDst[tileIndex];
                            tileDst = tilesSrc[tileIndex].transform(cameraToWorld, cameraToWorldInvTrans);

                            SimpleRayFrustum& simpleFrustum = simpleTileFrusta[tileIndex];
                            for (int n = 0; n < RayPacketFrustum3D::pointCount; n++) {
                                simpleFrustum.origins[n].x = tileDst.pointOrigin[n].x;
                                simpleFrustum.origins[n].y = tileDst.pointOrigin[n].y;
                                simpleFrustum.origins[n].z = tileDst.pointOrigin[n].z;

                                simpleFrustum.directions[n].x = tileDst.pointDir[n].x;
                                simpleFrustum.directions[n].y = tileDst.pointDir[n].y;
                                simpleFrustum.directions[n].z = tileDst.pointDir[n].z;
                            }
                        }
                    }
                };

                enum { maxTasks = 4096 };
                enum { blocksPerThread = 16 };
                uint32_t numTasks = (blockCount + blocksPerThread - 1) / blocksPerThread;
                assert(numTasks <= maxTasks);
                numTasks = min<uint32_t>(maxTasks, numTasks);

                std::future<void> taskResults[maxTasks];
                for (uint32_t i = 0; i < numTasks; ++i) {
                    uint32_t startBlock = min(blockCount, i * blocksPerThread);
                    uint32_t endBlock = min(blockCount, (i + 1) * blocksPerThread);

                    taskResults[i] = _threadPool->addTask(blockTransformTask, startBlock, endBlock);
                }
                for (uint32_t i = 0; i < numTasks; ++i) {
                    taskResults[i].get();
                }
            }

            gpuCamera->updatePerFrame(camera->getTranslation(), camera->getForward(), camera->getSampleToCamera(),
                                      cameraToWorld);

            BlockInfo blockInfo;
            blockInfo.blockFrusta = camera->_blockFrustaTransformed;
            blockInfo.tileFrusta = camera->_tileFrustaTransformed;

#if DUMP_SCENE_AND_RAYS
            static bool dumped = false;
            if (!dumped) {
                Timer timer;

                dumpSceneToOFF("scene_dump.off", _models);

                std::vector<SimpleRay> rays;
                gpuCamera->dumpRays(rays, DUMP_IN_SCANLINE_ORDER == 1);
                dumpRaysToRFF("ray_dump.rff", rays);

                double dumpTime = timer.get();
                printf("dump time %f\n", dumpTime);

                dumped = true;
            }
#endif

            buildTileTriangleLists(blockInfo, streamed);

            // end filling data for the GPU
            gpuCamera->streamedDataUnlock();

            gpuCamera->intersectShadeResolve(_gpuContext->sceneState);
            gpuCamera->remap();

            // size_t totalTris = _bvhScene.triangles.size;
            // size_t trisIntersected = camera->tileCullInfo.triIndexCount;
            // printf("Intersection Ratio %f: %d/%d\n", (float)trisIntersected / totalTris, trisIntersected, totalTris);
        }
    }
#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    // Copy the results to the camera's DX texture
    for (auto& camera : _cameras) {
        if (!camera->getEnabled())
            continue;
        GPUCamera* gpuCamera = camera->_gpuCamera;

#if OUTPUT_MODE == OUTPUT_MODE_3D_API
        if (camera->_renderTarget.isHardwareRenderTarget()) {
            gpuCamera->copyImageToBoundTexture();
        } else {
            gpuCamera->copyImageToCPU((uint32_t*)camera->_renderTarget.data, camera->_renderTarget.width,
                                      camera->_renderTarget.height, uint32_t(camera->_renderTarget.stride));
        }
#endif
    }
    _gpuContext->interopUnmapResources();
#endif
}

// TODO(anankervis): merge into a single render path
void Raycaster::renderFoveatedPolarSpaceCudaReconstruct() {
    polarSpaceFoveatedSetup(this);
#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    for (auto& camera : _cameras) {
        if (!camera->getEnabled())
            continue;
        GPUCamera* gpuCamera = camera->_gpuCamera;

        if (camera->_renderTarget.isHardwareRenderTarget() && camera->_newHardwareTarget) {
            gpuCamera->bindTexture(*_gpuContext, camera->_renderTarget);
            camera->_newHardwareTarget = false;
        }
    }
#endif
    for (auto& camera : _cameras) {
        if (!camera->getEnabled())
            continue;
        GPUCamera* gpuCamera = camera->_gpuCamera;

        const SampleData& sampleData = camera->getSampleData();
        const vector3& eyeDirection = camera->getEyeDir();

        vector2 jitter = camera->_frameJitters[camera->_frameCount % camera->_frameJitters.size()];
        gpuCamera->setCameraJitter(vector2(jitter.x * 0.5f + 0.5f, jitter.y * 0.5f + 0.5f));
        ++camera->_frameCount;

        matrix3x3 cameraToSample = invert(camera->getSampleToCamera());
        matrix3x3 eyeToCamera = matrix3x3::rotationFromZAxis(-eyeDirection);
        matrix4x4 eyeToWorld = camera->getCameraToWorld() * matrix4x4(eyeToCamera);
        // TODO: principled derivation of magic constant
        FloatRect cullRect = expandRect(sampleData.sampleBounds, 0.15f);

        gpuCamera->updatePerFrameFoveatedData(cullRect, cameraToSample, eyeToCamera, eyeToWorld);

        auto blockCount = (camera->_foveatedSampleData.eyeSpaceSamples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        camera->_foveatedSampleData.blockCount = blockCount;
    }

#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    _gpuContext->interopMapResources();
#endif
    {
        for (auto& camera : _cameras) {
            if (!camera->getEnabled())
                continue;
            GPUCamera* gpuCamera = camera->_gpuCamera;

            gpuCamera->acquireTileCullData(camera->_foveatedSampleData.simpleTileFrusta.data(),
                                           camera->_foveatedSampleData.simpleBlockFrusta.data());
            uint32_t blockCount = uint32_t(camera->_foveatedSampleData.simpleBlockFrusta.size());
            uint32_t tileCount = uint32_t(camera->_foveatedSampleData.simpleTileFrusta.size());

            // begin filling data for the GPU
            Camera_StreamedData* streamed = gpuCamera->streamedDataLock(tileCount);

            for (uint32_t i = 0; i < blockCount; ++i) {
                camera->_foveatedSampleData.samples.blockFrusta3D[i] =
                    RayPacketFrustum3D(camera->_foveatedSampleData.simpleBlockFrusta[i]);
            }

            SimpleRayFrustum* simpleTileFrusta = streamed->tileFrusta3D.dataHost();
            for (uint32_t i = 0; i < tileCount; ++i) {
                camera->_foveatedSampleData.samples.tileFrusta3D[i] =
                    RayPacketFrustum3D(camera->_foveatedSampleData.simpleTileFrusta[i]);

                // TODO(anankervis): avoid the GPU -> CPU -> CPU -> GPU copy of the SimpleRayFrustum
                simpleTileFrusta[i] = camera->_foveatedSampleData.simpleTileFrusta[i];
            }

            gpuCamera->updatePerFrame(camera->getTranslation(), camera->getForward(), camera->getSampleToCamera(),
                                      camera->getCameraToWorld());

            BlockInfo blockInfo;
            blockInfo.blockFrusta = ArrayView<const RayPacketFrustum3D>(
                camera->_foveatedSampleData.samples.blockFrusta3D.data(), camera->_foveatedSampleData.blockCount);
            blockInfo.tileFrusta =
                ArrayView<const RayPacketFrustum3D>(camera->_foveatedSampleData.samples.tileFrusta3D.data(),
                                                    camera->_foveatedSampleData.blockCount * TILES_PER_BLOCK);

            // Uncomment to get stats on how good the block/tile clustering is working
            // debugPrintTileCost(camera->polarFoveatedSampleData.samples, camera->polarFoveatedSampleData.blockCount);
            buildTileTriangleLists(blockInfo, streamed);

            // size_t totalTris = _bvhScene.triangles.size;
            // size_t trisIntersected = camera->tileCullInfo.triIndexCount;
            // printf("Intersection Ratio %f: %d/%d\n", (float)trisIntersected/totalTris, trisIntersected, totalTris);

            // end filling data for the GPU
            gpuCamera->streamedDataUnlock();

            gpuCamera->intersectShadeResolve(_gpuContext->sceneState);
            gpuCamera->remapPolarFoveated();

            const vector3& eyeDirection = camera->getEyeDir();
            matrix3x3 eyeToCamera = matrix3x3::rotationFromZAxis(-eyeDirection);
            matrix3x3 sampleToEye = invert(eyeToCamera) * camera->getSampleToCamera();

            matrix4x4 eyeToWorld = camera->getCameraToWorld() * matrix4x4(eyeToCamera);
            matrix4x4 eyeToEyePrevious = camera->_worldToEyePrevious * eyeToWorld;

            // Handles Temporal Filtering if necessary
            gpuCamera->foveatedPolarToScreenSpace(eyeToEyePrevious, camera->_eyePreviousToSamplePrevious, sampleToEye);

            camera->_worldToEyePrevious = invert(eyeToWorld);
            camera->_eyePreviousToSamplePrevious = invert(sampleToEye);
        }
    }
#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    // Copy the results to the camera's DX texture
    for (auto& camera : _cameras) {
        if (!camera->getEnabled())
            continue;
        GPUCamera* gpuCamera = camera->_gpuCamera;

        if (camera->_renderTarget.isHardwareRenderTarget()) {
            gpuCamera->copyImageToBoundTexture();
        } else {
            gpuCamera->copyImageToCPU((uint32_t*)camera->_renderTarget.data, camera->_renderTarget.width,
                                      camera->_renderTarget.height, uint32_t(camera->_renderTarget.stride));
        }
    }
    _gpuContext->interopUnmapResources();
#endif
}

} // namespace hvvr
