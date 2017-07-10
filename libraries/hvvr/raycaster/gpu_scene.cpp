/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "cuda_raycaster.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "gpu_scene_state.h"
#include "light.h"
#include "memory_helpers.h"
#include "raycaster.h"


namespace hvvr {

void UpdateSceneGeometry(const Raycaster& raycaster) {
    GPUSceneState& gpuSceneState = gGPUContext->sceneState;
    gpuSceneState.setGeometry(raycaster);
}

void AnimateScene(const matrix4x4* modelToWorld, size_t modelToWorldCount) {
    GPUSceneState& gpuSceneState = gGPUContext->sceneState;
    gpuSceneState.animate(modelToWorld, modelToWorldCount);
}

bool FetchUpdatedBVH(BVHNode* dstNodes) {
    GPUSceneState& gpuSceneState = gGPUContext->sceneState;
    return gpuSceneState.fetchUpdatedBVH(dstNodes);
}

void UpdateLighting(const Raycaster& raycaster) {
    GPUSceneState& sceneState = gGPUContext->sceneState;

    LightingEnvironment& env = sceneState.lightingEnvironment;
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

void UpdateMaterials(SimpleMaterial* materials, size_t materialCount) {
    static_assert(sizeof(SimpleMaterial) == sizeof(SimpleMaterial),
                  "GPU and CPU SimpleMaterials mismatched");
    gGPUContext->sceneState.materials = GPUBuffer<SimpleMaterial>(
        (SimpleMaterial*)materials, ((SimpleMaterial*)materials) + materialCount);
}

} // namespace hvvr
