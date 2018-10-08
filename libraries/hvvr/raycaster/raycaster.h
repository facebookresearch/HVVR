#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dynamic_array.h"
#include "graphics_types.h"
#include "raycaster_spec.h"

#include <memory>
#include <string>
#include <vector>

struct PrecomputedTriangleShade;
struct BVHNode;
struct RayHierarchy;
struct Camera_StreamedData;
struct RayPacketFrustum3D;

namespace hvvr {

class ThreadPool;
class Camera;
class Texture;
class Light;
class Model;
struct MeshData;
struct TextureData;
struct LightUnion;
struct SimpleMaterial;
class GPUContext;

class Raycaster {
    friend class GPUSceneState;
    // TODO(anankervis): remove
    friend void polarSpaceFoveatedSetup(Raycaster* raycaster);

public:
    Raycaster(const RayCasterSpecification& spec);
    ~Raycaster();

    //// scene management
    Camera* createCamera(const FloatRect& viewport, float apertureRadius);
    void destroyCamera(Camera* camera);

    Texture* createTexture(const TextureData& textureData);
    void destroyTexture(Texture* texture);

    Light* createLight(const LightUnion& lightUnion);
    void destroyLight(Light* light);

    Model* createModel(MeshData&& meshData);
    Model* createModel(const MeshData& meshData);
    void destroyModel(Model* model);

    void setSceneDirty();

    //// rendering
    void render(double elapsedTime);

    void reinit(RayCasterGPUMode mode);

protected:
    Raycaster(const Raycaster&) = delete;
    Raycaster(Raycaster&&) = delete;
    Raycaster& operator=(const Raycaster&) = delete;
    Raycaster& operator=(Raycaster&&) = delete;

    RayCasterSpecification _spec;
    std::unique_ptr<ThreadPool> _threadPool;

    //// scene management
    std::vector<std::unique_ptr<Camera>> _cameras;
    std::vector<std::unique_ptr<Texture>> _textures;
    std::vector<std::unique_ptr<Light>> _lights;
    std::vector<std::unique_ptr<Model>> _models;

    // TODO(anankervis): remove me, calculate when uploading data to GPU instead
    std::vector<PrecomputedTriangleShade> _trianglesShade;
    DynamicArray<BVHNode> _nodes;
    std::vector<SimpleMaterial> _materials;
    uint32_t _vertexCount;

    bool _sceneDirty;

    //// GPU
    std::unique_ptr<GPUContext> _gpuContext;

    void updateScene(double elapsedTime);
    void buildScene();
    void uploadScene();
    void cleanupScene();

    void setupAllRenderTargets();
    void blitAllRenderTargets();

    void interopMapResources();
    void interopUnmapResources();


    //// rendering
    void renderCameraGPUIntersectAndReconstructDeferredMSAAResolve(std::unique_ptr<hvvr::Camera>& camera);
    void renderGPUIntersectAndReconstructDeferredMSAAResolve();
    void renderFoveatedPolarSpaceCudaReconstruct();
    void renderCamera(std::unique_ptr<hvvr::Camera>& camera);

    // traverse BVH and generate lists of triangles to intersect on the GPU
    void buildTileTriangleLists(const RayHierarchy& rayHierarchy, Camera_StreamedData* streamed);
    void transformHierarchyCameraToWorld(std::unique_ptr<hvvr::Camera>& camera,
                                         const RayPacketFrustum3D* tilesSrc,
                                         const RayPacketFrustum3D* blocksSrc,
                                         uint32_t blockCount,
                                         Camera_StreamedData* streamed);
};

} // namespace hvvr
