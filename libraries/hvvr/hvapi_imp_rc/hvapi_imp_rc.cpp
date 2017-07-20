/**
* Copyright (c) 2017-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include "../hvapi/hvapi.h"
#include "raycaster.h"

#include <vector>

#pragma warning(disable: 4100) // unreferenced formal parameter

extern "C" {

struct _hvInstance {
};
typedef _hvInstance* hvInstance;

struct _hvScene {
    std::vector<hvInstance> instances;
    std::vector<hvInstanceID> instanceFreeList;

    ~_hvScene() {
        for (hvInstance instance : instances) {
            delete instance;
        }
        instances.clear();
        instanceFreeList.clear();
    }
};

struct _hvMesh {
};

struct _hvRayBatch {
};

struct _hvRayGenerator {
};

struct _hvRayHits {
};

struct _hvHitCallback {
};

class hvContextState {
public:
    hvError lastError;
    hvErrorCallback errorCallback;

    std::vector<hvScene> scenes;
    std::vector<hvMesh> meshes;
    std::vector<hvRayBatch> rayBatches;
    std::vector<hvRayGenerator> rayGenerators;
    std::vector<hvRayHits> rayHits;
    std::vector<hvHitCallback> hitCallbacks;

    hvvr::Raycaster raycaster;

    hvContextState() : lastError(HV_ERROR_NONE), errorCallback(nullptr), raycaster(RayCasterSpecification()) {
    }

    ~hvContextState() {
        for (hvScene scene : scenes) {
            delete scene;
        }
        scenes.clear();

        for (hvMesh mesh : meshes) {
            delete mesh;
        }
        meshes.clear();

        for (hvRayBatch rayBatch : rayBatches) {
            delete rayBatch;
        }
        rayBatches.clear();

        for (hvRayGenerator rayGenerator : rayGenerators) {
            delete rayGenerator;
        }
        rayGenerators.clear();

        for (hvRayHits rayHit : rayHits) {
            delete rayHit;
        }
        rayHits.clear();

        for (hvHitCallback hitCallback : hitCallbacks) {
            delete hitCallback;
        }
        hitCallbacks.clear();
    }

    void setError(hvError error, const char* errorStr) {
        lastError = error;
        if (lastError != HV_ERROR_NONE && errorCallback) {
            errorCallback(error, errorStr);
        }
    }
};
static hvContextState* getContextState() {
    static hvContextState contextState;
    return &contextState;
}

hvError HVAPI hvGetLastError() {
    hvContextState& context = *getContextState();

    return context.lastError;
}

void HVAPI hvSetErrorCallback(hvErrorCallback errorCallback) {
    hvContextState& context = *getContextState();

    context.errorCallback = errorCallback;
}


hvScene HVAPI hvCreateScene(
    uint32_t flags) {
    hvContextState& context = *getContextState();

    if (context.scenes.size() != 0) {
        context.setError(HV_ERROR_NOT_IMPLEMENTED, "Only one scene at a time is currently supported");
        return nullptr;
    }

    hvScene newScene = new _hvScene;
    context.scenes.push_back(newScene);
    return newScene;
}

void HVAPI hvDestroyScene(hvScene scene) {
    hvContextState& context = *getContextState();

    auto it = std::find(context.scenes.begin(), context.scenes.end(), scene);
    if (it == context.scenes.end()) {
        context.setError(HV_ERROR_INVALID_ARG, "Invalid scene");
        return;
    }

    delete *it;
    context.scenes.erase(it);
}


hvMesh HVAPI hvCreateMesh(
    const hvFloat3* positions, uint32_t vertexCount, uint32_t vertexStrideBytes,
    const uint32_t* indices, uint32_t triangleCount,
    uint32_t flags) {
    hvContextState& context = *getContextState();
    
    hvMesh newMesh = new _hvMesh;
    context.meshes.push_back(newMesh);
    return newMesh;
}

void HVAPI hvDestroyMesh(hvMesh mesh) {
    hvContextState& context = *getContextState();

    auto it = std::find(context.meshes.begin(), context.meshes.end(), mesh);
    if (it == context.meshes.end()) {
        context.setError(HV_ERROR_INVALID_ARG, "Invalid mesh");
        return;
    }

    delete *it;
    context.meshes.erase(it);
}

void HVAPI hvUpdateMeshVertices(
    hvMesh mesh, const hvFloat3* positions, uint32_t vertexStrideBytes) {
    hvContextState& context = *getContextState();

    context.setError(HV_ERROR_NOT_IMPLEMENTED, "hvUpdateMeshVertices not implemented");
}


hvInstanceID HVAPI hvCreateInstance(
    hvScene scene, hvMesh meshToAttach, const hvTransform* transform, uint32_t flags) {
    hvContextState& context = *getContextState();

    hvInstanceID instanceID = HV_INVALID_INSTANCE;

    if (!scene->instanceFreeList.empty()) {
        instanceID = scene->instanceFreeList.back();
        scene->instanceFreeList.erase(scene->instanceFreeList.end() - 1);
    } else {
        if (scene->instances.size() >= HV_INVALID_INSTANCE) {
            context.setError(HV_ERROR_INVALID_OP, "Attempted to create too many instances in the scene");
            return HV_INVALID_INSTANCE;
        }

        instanceID = hvInstanceID(scene->instances.size());
        scene->instances.resize(instanceID + 1);
    }

    hvInstance newInstance = new _hvInstance;
    scene->instances[instanceID] = newInstance;
    return instanceID;
}

void HVAPI hvDestroyInstance(hvScene scene, hvInstanceID instanceID) {
    hvContextState& context = *getContextState();

    if (instanceID >= scene->instances.size()) {
        context.setError(HV_ERROR_INVALID_OP, "Invalid instanceID");
    }

    delete scene->instances[instanceID];
    scene->instances[instanceID] = nullptr;
    scene->instanceFreeList.push_back(instanceID);
}

void HVAPI hvUpdateInstanceTransform(
    hvScene scene, hvInstanceID instanceID, const hvTransform* transform) {
    hvContextState& context = *getContextState();

    context.setError(HV_ERROR_NOT_IMPLEMENTED, "hvUpdateInstanceTransform not implemented");
}


hvRayGenerator HVAPI hvCreateRayGenerator(
    const hvRayGenConfig* rayGenConfig, uint32_t flags) {
    hvContextState& context = *getContextState();

    hvRayGenerator newRayGenerator = new _hvRayGenerator;
    context.rayGenerators.push_back(newRayGenerator);
    return newRayGenerator;
}

void HVAPI hvDestroyRayGenerator(hvRayGenerator rayGenerator) {
    hvContextState& context = *getContextState();

    auto it = std::find(context.rayGenerators.begin(), context.rayGenerators.end(), rayGenerator);
    if (it == context.rayGenerators.end()) {
        context.setError(HV_ERROR_INVALID_ARG, "Invalid rayGenerator");
        return;
    }

    delete *it;
    context.rayGenerators.erase(it);
}


hvRayBatch HVAPI hvCreateRayBatch(
    hvRayGenerator rayGenerator, const hvRayGenParams* rayGenParams,
    uint32_t rayCount, uint32_t subsampleCount, uint32_t flags) {
    hvContextState& context = *getContextState();

    hvRayBatch newRayBatch = new _hvRayBatch;
    context.rayBatches.push_back(newRayBatch);
    return newRayBatch;
}

void HVAPI hvDestroyRayBatch(hvRayBatch rayBatch) {
    hvContextState& context = *getContextState();

    auto it = std::find(context.rayBatches.begin(), context.rayBatches.end(), rayBatch);
    if (it == context.rayBatches.end()) {
        context.setError(HV_ERROR_INVALID_ARG, "Invalid rayBatch");
        return;
    }

    delete *it;
    context.rayBatches.erase(it);
}


hvRayHits HVAPI hvCreateRayHits(
    hvRayBatch rays, uint32_t flags) {
    hvContextState& context = *getContextState();

    hvRayHits newRayHits = new _hvRayHits;
    context.rayHits.push_back(newRayHits);
    return newRayHits;
}

void HVAPI hvDestroyRayHits(hvRayHits rayHits) {
    hvContextState& context = *getContextState();

    auto it = std::find(context.rayHits.begin(), context.rayHits.end(), rayHits);
    if (it == context.rayHits.end()) {
        context.setError(HV_ERROR_INVALID_ARG, "Invalid rayHits");
        return;
    }

    delete *it;
    context.rayHits.erase(it);
}


hvHitCallback HVAPI hvCreateHitCallback(
    const hvHitCallbackConfig* hitCallbackConfig, uint32_t flags) {
    hvContextState& context = *getContextState();

    hvHitCallback newHitCallback = new _hvHitCallback;
    context.hitCallbacks.push_back(newHitCallback);
    return newHitCallback;
}

void HVAPI hvDestroyHitCallback(hvHitCallback hitCallback) {
    hvContextState& context = *getContextState();

    auto it = std::find(context.hitCallbacks.begin(), context.hitCallbacks.end(), hitCallback);
    if (it == context.hitCallbacks.end()) {
        context.setError(HV_ERROR_INVALID_ARG, "Invalid hitCallback");
        return;
    }

    delete *it;
    context.hitCallbacks.erase(it);
}


void HVAPI hvProcessHits(
    hvRayHits hits, hvHitCallback shadingCallback, hvRayBatch* outRays, uint32_t flags) {
    hvContextState& context = *getContextState();

    context.setError(HV_ERROR_NOT_IMPLEMENTED, "hvProcessHits not implemented");
}


void HVAPI hvTrace(
    hvScene scene, hvRayBatch rays, hvHitCallback shadingCallback, uint32_t flags) {
    hvContextState& context = *getContextState();

    context.setError(HV_ERROR_NOT_IMPLEMENTED, "hvTrace not implemented");
}

void HVAPI hvTraceDeferred(
    hvScene scene, hvRayBatch rays, hvRayHits outHits, uint32_t flags) {
    hvContextState& context = *getContextState();

    context.setError(HV_ERROR_NOT_IMPLEMENTED, "hvTraceDeferred not implemented");
}

} // extern "C"
