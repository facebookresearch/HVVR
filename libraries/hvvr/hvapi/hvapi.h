#ifndef HVAPI_H
#define HVAPI_H

/**
* Copyright (c) 2017-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
# define HVAPI __stdcall
#else
# define HVAPI
#endif

// opaque types
typedef struct _hvMesh* hvMesh;
typedef struct _hvScene* hvScene;
typedef struct _hvRayBatch* hvRayBatch;
typedef struct _hvRayGenerator* hvRayGenerator;
typedef struct _hvRayHits* hvRayHits;
typedef struct _hvHitCallback* hvHitCallback;

// exposed types
typedef uint32_t hvInstance;

typedef struct _hvFloat2 {
    float x, y;
} hvFloat2;

typedef struct _hvFloat3 {
    float x, y, z;
} hvFloat3;

typedef struct _hvQuaternion {
    float x, y, z, w;
} hvQuaternion;

typedef struct _hvTransform {
    hvQuaternion rotation;
    hvFloat3 position;
    float scale;
} hvTransform;

typedef struct _hvHit {
    hvInstance instance;
    uint32_t triangleIndex;
} hvHit;

// ray batches share an aperture, so no need for an explicit per-ray origin
typedef struct _hvRay {
    hvFloat3 focalPoint;

    // ray footprint
    hvFloat3 focalDeltaMajor;
    hvFloat3 focalDeltaMinor;

    // What is this used for? We'll need to vary the aperture offset per-subsample, and also
    // rotate it per-ray, but I'm not sure if there's a need for a per-ray 2D offset.
    hvFloat2 apertureOffset;
} hvRay;


typedef enum _hvError {
    HV_ERROR_NONE            = 0,
    HV_ERROR_NOT_IMPLEMENTED = 1,
    HV_ERROR_INVALID_ARG     = 2,
    HV_ERROR_INVALID_OP      = 3,
} hvError;

typedef void(*hvErrorCallback)(hvError error, const char* errorStr);

hvError HVAPI hvGetLastError();
void HVAPI hvSetErrorCallback(hvErrorCallback errorCallback);


typedef enum _hvCreateSceneFlags {
    HV_CREATE_SCENE_FLAGS_NONE = 0x0,
} hvCreateSceneFlags;

hvScene HVAPI hvCreateScene(
    uint32_t flags);

void HVAPI hvDestroyScene(hvScene scene);


typedef enum _hvCreateMeshFlags {
    HV_CREATE_MESH_FLAGS_NONE    = 0x0,
    HV_CREATE_MESH_FLAGS_DYNAMIC = 0x1, // vertex positions (not indices?) can change after creation
} hvCreateMeshFlags;

hvMesh HVAPI hvCreateMesh(
    const hvFloat3* positions, uint32_t vertexCount, uint32_t vertexStrideBytes,
    const uint32_t* indices, uint32_t triangleCount,
    uint32_t flags);

void HVAPI hvDestroyMesh(hvMesh mesh);

// HV_ERROR_INVALID_OP if mesh was not created with HV_CREATE_MESH_FLAGS_DYNAMIC
void HVAPI hvUpdateMeshVertices(
    hvMesh mesh, const hvFloat3* positions, uint32_t vertexStrideBytes);


typedef enum _hvCreateInstanceFlags {
    HV_CREATE_INSTANCE_FLAGS_NONE    = 0x0,
    HV_CREATE_INSTANCE_FLAGS_DYNAMIC = 0x1, // instance transform can change after creation
} hvCreateInstanceFlags;

// do we specify instance transform as quaternion+pos+scale, or as a 4x3 matrix, or?
hvInstance HVAPI hvCreateInstance(
    hvScene targetScene, hvMesh meshToAttach, const hvTransform* transform, uint32_t flags);

void HVAPI hvDestroyInstance(hvInstance instance);

// HV_ERROR_INVALID_OP if instance was not created with HV_CREATE_INSTANCE_FLAGS_DYNAMIC
void HVAPI hvUpdateInstanceTransform(
    hvInstance instance, const hvTransform* transform);


typedef enum _hvCreateRayGeneratorFlags {
    HV_CREATE_RAY_GENERATOR_FLAGS_NONE = 0x0,
} hvCreateRayGeneratorFlags;

typedef enum _hvRayGenType {
    HV_RAY_GEN_TYPE_HOST_BUFFER   = 0, // pull rays from an array in CPU memory
/*
    HV_RAY_GEN_TYPE_DEVICE_BUFFER = 1, // pull rays from an array in GPU memory
    HV_RAY_GEN_TYPE_HOST_FUNC     = 2, // procedurally generate rays on the CPU
    HV_RAY_GEN_TYPE_DEVICE_FUNC   = 3, // procedurally generate rays on the GPU
    HV_RAY_GEN_TYPE_SUBDIV_SURF   = 4, // fixed-function subdivision surface
    // rays are produced by a shading hit callback
    // the rayCount parameter to hvCreateRayBatch must be zero
    HV_RAY_GEN_TYPE_HIT_CALLBACK  = 5,
*/
} hvRayGenType;

typedef struct _hvRayGenConfig {
    hvRayGenType rayGenType;

    union {
        struct {
        } hostBuffer;
/*
        struct {
        } deviceBuffer;

        struct {
        } hostFunc;

        struct {
        } deviceFunc;

        struct {
        } subdivSurf;

        struct {
            // max number of rays to reserve space for
            // rays in excess of this amount will be discarded
            uint32_t maxRays;
        } hitCallback;
*/
    };
} hvRayGenConfig;

hvRayGenerator HVAPI hvCreateRayGenerator(
    const hvRayGenConfig* rayGenConfig, uint32_t flags);

void HVAPI hvDestroyRayGenerator(hvRayGenerator rayGenerator);


typedef enum _hvCreateRayBatchFlags {
    HV_CREATE_RAY_BATCH_FLAGS_NONE                    = 0x0,
    // Do we need this? Outside of shadows, wouldn't the rays change every frame due to camera motion?
    // this batch of rays changes frequently
    HV_CREATE_RAY_BATCH_FLAGS_DYNAMIC                 = 0x1,
    // all rays in the batch have a common origin (pinhole camera)
    HV_CREATE_RAY_BATCH_FLAGS_COMMON_ORIGIN           = 0x2,
    // all rays in the batch are on a regular grid, requires HV_CREATE_RAY_BATCH_FLAGS_COMMON_ORIGIN
    HV_CREATE_RAY_BATCH_FLAGS_REGULAR_GRID            = 0x4,
    // source rays are pre-grouped into coherent clusters in a two-level hierarchy:
    // 128 rays per tile
    // 64 tiles per block
    HV_CREATE_RAY_BATCH_FLAGS_ORDERED_TILE128_BLOCK64 = 0x8,
} hvCreateRayBatchFlags;

typedef struct _hvRayGenParams {
    hvFloat3 origin;
    hvFloat3 apertureTangent;
    hvFloat3 apertureBiTangent;

    union {
        struct {
            const hvRay* raysSrc;
        } hostBuffer;
/*
        struct {
        } deviceBuffer;

        struct {
        } hostFunc;

        struct {
        } deviceFunc;

        struct {
        } subdivSurf;

        struct {
        } hitCallback;
*/
    };
} hvRayGenParams;

hvRayBatch HVAPI hvCreateRayBatch(
    hvRayGenerator rayGenerator, const hvRayGenParams* rayGenParams,
    uint32_t rayCount, uint32_t subsampleCount, uint32_t flags);

void HVAPI hvDestroyRayBatch(hvRayBatch rayBatch);


typedef enum _hvCreateRayHitsFlags {
    HV_CREATE_RAY_HITS_FLAGS_NONE          = 0x0,
} hvCreateRayHitsFlags;

hvRayHits HVAPI hvCreateRayHits(
    hvRayBatch rays, uint32_t flags);

void HVAPI hvDestroyRayHits(hvRayHits rayHits);


typedef enum _hvCreateHitCallbackFlags {
    HV_CREATE_HIT_CALLBACK_FLAGS_NONE       = 0x0,
    HV_CREATE_HIT_CALLBACK_FLAGS_EMITS_RAYS = 0x1, // the callback can emit new rays
} hvCreateHitCallbackFlags;

typedef enum _hvHitCallbackType {
    HV_HIT_CALLBACK_TYPE_HOST_FUNC     = 0, // procedurally shades ray hits on the CPU
/*
    HV_HIT_CALLBACK_TYPE_DEVICE_FUNC   = 1, // procedurally shades ray hits on the GPU
    HV_HIT_CALLBACK_TYPE_PREDEFINED    = 2, // implementation-provided shading function
*/
} hvHitCallbackType;

typedef struct _hvShadingFuncConfig {
    hvHitCallbackType hitCallbackType;

    union {
        struct {
            // function pointer
        } hostFunc;
/*
        struct {
        } deviceFunc;

        struct {
            // which predefined function?
        } predefinedFunc;
*/
    };
} hvHitCallbackConfig;

hvHitCallback HVAPI hvCreateHitCallback(
    const hvHitCallbackConfig* hitCallbackConfig, uint32_t flags);

void HVAPI hvDestroyHitCallback(hvHitCallback hitCallback);


typedef enum _hvProcessHitsFlags {
    HV_PROCESS_HITS_FLAGS_NONE        = 0x0,
    // append newly generated rays to the existing ray batch, instead of resetting the ray count to zero
    HV_PROCESS_HITS_FLAGS_APPEND_RAYS = 0x1,
} hvProcessHitsFlags;

// outRays is optional, and if supplied must be paired with an hvRayBatch created with
// HV_RAY_GEN_TYPE_HIT_CALLBACK and an hvHitCallback created with HV_CREATE_HIT_CALLBACK_FLAGS_EMITS_RAYS
void HVAPI hvProcessHits(
    hvRayHits hits, hvHitCallback shadingCallback, hvRayBatch* outRays, uint32_t flags);


typedef enum _hvTraceFlags {
    HV_TRACE_FLAGS_NONE    = 0x0,
    // terminate traversal on any hit instead of finding closest hit to the ray (for shadow/connection rays)
    HV_TRACE_FLAGS_ANY_HIT = 0x1,
} hvTraceFlags;

// If shadingCallback was created with HV_CREATE_HIT_CALLBACK_FLAGS_EMITS_RAYS, trace will
// place the intermediate rays into implementation-defined storage and continue tracing
// until no additional rays are generated.
// Should we have a maxIterations or maxRays param to bound runtime and avoid infinite loops?
void HVAPI hvTrace(
    hvScene scene, hvRayBatch rays, hvHitCallback shadingCallback, uint32_t flags);
void HVAPI hvTraceDeferred(
    hvScene scene, hvRayBatch rays, hvRayBatch outHits, uint32_t flags);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HVAPI_H
