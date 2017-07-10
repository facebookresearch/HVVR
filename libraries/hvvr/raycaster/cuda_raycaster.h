#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "material.h"
#include "raycaster_spec.h"
#include "samples.h"
#include "vector_math.h"
#include "graphics_types.h"


struct ID3D11Buffer;
struct BVHNode;

namespace hvvr {

class Camera;
class Raycaster;
struct GPUCamera;
struct PrecomputedDirectionSample;

// TODO(anankervis): find a new home for this
inline PixelFormat outputModeToPixelFormat(RaycasterOutputMode mode) {
    (void)mode;
    return PixelFormat::RGBA8_SRGB;
}

void CalculatePerFrameFoveatedData(Camera* cameraPtr,
                                   const FloatRect& sampleBounds,
                                   const matrix3x3& cameraToSamplesMatrix,
                                   const matrix3x3& eyeToCameraMatrix,
                                   const matrix4x4& eyeToWorldMatrix);

// Assumes CalculatePerFrameFoveatedData has previously been called
void AcquireTileCullInformation(Camera* cameraPtr,
                                SimpleRayFrustum* tileFrusta,
                                SimpleRayFrustum* blockFrusta);

// Copies the PrecomputedDirectionSample's onto the GPU
void UpdateEyeSpaceFoveatedSamples(Camera* cameraPtr, const hvvr::ArrayView<PrecomputedDirectionSample> precomputedDirectionalSamples);

// At the end, the camera corresponding to \param cameraName holds a pointer to the final image, stored in a device
// memory
// Call CopyImageToCPU or CopyImageToBoundTexture to get the results on the CPU or in the previously bound texture,
// respectively

void DeferredMSAAIntersect(Camera* cameraPtr);
int DeferredMSAAIntersectAndRemap(Camera* cameraPtr);
void PolarFoveatedRemap(Camera* cameraPtr);

void ConvertFromFoveatedPolarToScreenSpace(Camera* cameraPtr,
                                           const matrix4x4& eyeSpaceToPreviousEyeSpaceMatrix,
                                           const matrix3x3& previousEyeSpaceToPreviousSampleSpaceMatrix,
                                           const matrix3x3& sampleSpaceToEyeSpaceMatrix);

GPUCamera* GetCamera(Camera* cameraPtr);

// \param sampleLocations is a flat array of float2s of the sample locations, blocked only by block
// \param sampleResultBuffer: if not null, bind this buffer as a cuda resource and write into it
void UpdateCameraConfig(Camera* cameraPtr,
                        RaycasterOutputMode outputMode,
                        int32_t* sampleRemap,
                        float* sampleLocations,
                        Sample::Extents* sampleExtents,
                        ThinLens lens,
                        uint32_t sampleCount,
                        uint32_t imageWidth,
                        uint32_t imageHeight,
                        uint32_t imageStride,
                        uint32_t splitColorSamples);

void UpdateCameraPerFrame(Camera* cameraPtr,
                          vector3 cameraPos,
                          vector3 cameraLookVector,
                          const matrix3x3& sampleToCamera,
                          const matrix4x4& cameraToWorld);

// Copies the existing device memory image to \param imageData
int CopyImageToCPU(
    Camera* cameraPtr, unsigned int* imageData, uint32_t imageWidth, uint32_t imageHeight, uint32_t imageStride);

bool CameraHasBoundTexture(Camera* cameraPtr);

// Binds \param texture to the camera corresponding to \param cameraName, so that it can be updated by
// CopyImageToBoundTexture
int BindTextureToCamera(Camera* cameraPtr, ImageResourceDescriptor texture);

// Copies the existing device memory image to the camera's bound dx11 texture
int CopyImageToBoundTexture(Camera* cameraPtr);

// if this is called, AnimateScene must also be called afterward to populate the transforms
void UpdateSceneGeometry(const Raycaster& raycaster);
void AnimateScene(const matrix4x4* modelToWorld, size_t modelToWorldCount);
// if an updated BVH is available on the GPU, returns true and copies the GPU data to the CPU pointer
// assumes size of dstNodes matches the number of nodes last passed into UpdateSceneGeometry
bool FetchUpdatedBVH(BVHNode* dstNodes);

void UpdateLighting(const Raycaster& raycaster);

// Call each frame before using any bound textures in DX
int UnmapGraphicsResources();

// Call each frame before running any cuda kernels that use bound textures/buffers
int MapGraphicsResources();

void RegisterPolarFoveatedSamples(Camera* cameraPtr,
                                  const std::vector<vector2ui>& polarRemapToPixel,
                                  float maxEccentricityRadians,
                                  const std::vector<float>& ringEccentricities,
                                  const std::vector<float>& eccentricityCoordinateMap,
                                  uint32_t samplesPerRing,
                                  uint32_t paddedSampleCount);

void SetCameraJitter(Camera* cameraPtr, vector2 jitter);

void UpdateMaterials(SimpleMaterial* materialData, size_t materialCount);

void CreateExplicitRayBuffer(Camera* cameraPtr, std::vector<SimpleRay>& rays);

bool Init();
int Cleanup();

} // namespace hvvr
