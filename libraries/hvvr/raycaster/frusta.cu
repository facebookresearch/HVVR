/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "constants_math.h"
#include "cuda_decl.h"
#include "foveated.h"
#include "frusta.h"
#include "gpu_buffer.h"
#include "gpu_camera.h"
#include "gpu_samples.h"
#include "kernel_constants.h"
#include "memory_helpers.h"
#include "sort.h"
#include "traversal.h"
#include "vector_math.h"
#include "warp_ops.h"


namespace hvvr {

void ComputeEyeSpaceFrusta(const GPUBuffer<DirectionalBeam>& dirSamples,
                           GPUBuffer<SimpleRayFrustum>& tileFrusta,
                           GPUBuffer<SimpleRayFrustum>& blockFrusta) {
    DynamicArray<DirectionalBeam> samples = makeDynamicArray(dirSamples);
    DynamicArray<SimpleRayFrustum> tFrusta = makeDynamicArray(tileFrusta);
    DynamicArray<SimpleRayFrustum> bFrusta = makeDynamicArray(blockFrusta);

    const bool checkFrustaAccuracy = false;
    const bool printStats = false;

    auto generateFrusta = [](DynamicArray<SimpleRayFrustum>& frusta, unsigned int frustaSampleCount,
                             const DynamicArray<DirectionalBeam>& samples, float slopFactor, int numOrientationsToTry) {
        auto toDir = [](const matrix3x3& rot, float u, float v) { return rot * normalize(vector3(u, v, 1.0f)); };
        for (int i = 0; i < frusta.size(); ++i) {
            int sBegin = i * frustaSampleCount;
            int sEnd = min((int)((i + 1) * frustaSampleCount), (int)samples.size());
            vector3 dominantDirection(0.0f);
            for (int s = sBegin; s < sEnd; ++s) {
                dominantDirection += samples[s].centerRay;
            }
            dominantDirection = normalize(dominantDirection);

            // Try several different orientations for the plane, pick the one that
            // gives the smallest bounding box in uv space
            matrix3x3 rot(matrix3x3::rotationFromZAxis(dominantDirection));
            float bestUVArea = INFINITY;
            matrix3x3 bestRot;
            vector2 bestMinUV = vector2(INFINITY);
            vector2 bestMaxUV = vector2(-INFINITY);
            for (int o = 0; o < numOrientationsToTry; ++o) {
                const float range = (Pi / 2.0f) * 0.8f;
                matrix3x3 currRot =
                    matrix3x3::axisAngle(vector3(0, 0, 1), (range * o / float(numOrientationsToTry)) - (range / 2.0f)) *
                    rot;
                matrix3x3 invCurrRot = invert(currRot);
                vector2 minUV = vector2(INFINITY);
                vector2 maxUV = vector2(-INFINITY);
                for (int s = sBegin; s < sEnd; ++s) {
                    vector3 v = invCurrRot * samples[s].centerRay;
                    vector2 uv = vector2(v.x / v.z, v.y / v.z);
                    // TODO: check math here
                    v = invCurrRot * (samples[s].du + samples[s].centerRay);
                    float uvRadius = length(uv - vector2(v.x / v.z, v.y / v.z));
                    v = invCurrRot * (samples[s].dv + samples[s].centerRay);
                    uvRadius = max(uvRadius, length(uv - vector2(v.x / v.z, v.y / v.z)));
                    // slop; TODO: is this necessary, or can we do something more principled?
                    uvRadius *= slopFactor;
                    minUV = min(minUV, uv - uvRadius);
                    maxUV = max(maxUV, uv + uvRadius);
                }
                float uvArea = (maxUV.x - minUV.x) * (maxUV.y - minUV.y);
                if (uvArea < bestUVArea) {
                    bestRot = currRot;
                    bestUVArea = uvArea;
                    bestMinUV = minUV;
                    bestMaxUV = maxUV;
                }
            }

            SimpleRayFrustum f;
            for (int o = 0; o < 4; ++o) {
                f.origins[o] = {0.0f, 0.0f, 0.0f};
            }
            f.directions[0] = toDir(bestRot, bestMinUV.x, bestMaxUV.y);
            f.directions[1] = toDir(bestRot, bestMaxUV.x, bestMaxUV.y);
            f.directions[2] = toDir(bestRot, bestMaxUV.x, bestMinUV.y);
            f.directions[3] = toDir(bestRot, bestMinUV.x, bestMinUV.y);

            if (printStats) {
                for (int o = 0; o < 4; ++o) {
                    printf("f[%d].directions[%d]: %f, %f, %f\n", i, o, f.directions[o].x, f.directions[o].y,
                           f.directions[o].z);
                }
                printf("f[%d].bestUVArea: %f\n", i, bestUVArea);
                printf("Dominant Direction: %f %f %f\n", dominantDirection.x, dominantDirection.y, dominantDirection.z);
            }
            if (checkFrustaAccuracy) {
                // Make sure all samples points are within the frustum...
                RayPacketFrustum3D checker(f);
                for (int s = sBegin; s < sEnd; ++s) {
                    auto C = samples[s].centerRay;
                    if (!checker.testPoint(C)) {
                        printf("TROUBLE: f[%d]: s[%d]:%f %f %f \n", i, s, C.x, C.y, C.z);
                    }
                }
            }

            frusta[i] = f;
        }
    };
    generateFrusta(tFrusta, TILE_SIZE, samples, 2.0f, 63);
    generateFrusta(bFrusta, BLOCK_SIZE, samples, 2.0f, 63);

    tileFrusta = makeGPUBuffer(tFrusta);
    blockFrusta = makeGPUBuffer(bFrusta);
}


CUDA_HOST_DEVICE_INL bool planeCullsFrustum(const Plane plane, const SimpleRayFrustum& frustum) {
    bool allout = true;
    for (int i = 0; i < 4; ++i) {
        allout = allout && dot(plane.normal, frustum.origins[i]) > plane.dist;
        // Extend rays far-out
        allout = allout && dot(plane.normal, frustum.origins[i] + frustum.directions[i] * 10000.0f) > plane.dist;
    }
    return allout;
}

CUDA_KERNEL void CalculateWorldSpaceFrustaKernel(SimpleRayFrustum* blockFrustaWS,
                                                 SimpleRayFrustum* tileFrustaWS,
                                                 SimpleRayFrustum* blockFrustaES,
                                                 SimpleRayFrustum* tileFrustaES,
                                                 matrix4x4 eyeToWorldMatrix,
                                                 FourPlanes cullPlanes,
                                                 uint32_t blockCount,
                                                 uint32_t tileCount) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < blockCount + tileCount) {
        SimpleRayFrustum *inPtr, *outPtr;
        if (index < blockCount) {
            inPtr = (&blockFrustaES[index]);
            outPtr = (&blockFrustaWS[index]);
        } else {
            inPtr = (&tileFrustaES[index - blockCount]);
            outPtr = (&tileFrustaWS[index - blockCount]);
        }
        bool culled = false;
        SimpleRayFrustum f;
        for (int i = 0; i < 4; ++i) {
            f.origins[i] = vector3(eyeToWorldMatrix * vector4((*inPtr).origins[i], 1.0f));
            // We do not handle non-uniform scaling (could use inverse transpose of eyeToWorld to do so)
            f.directions[i] = normalize(vector3(eyeToWorldMatrix * vector4((*inPtr).directions[i], 0.0f)));
        }
        for (int i = 0; i < 4; ++i) {
            Plane p = cullPlanes.data[i];
            culled = culled || planeCullsFrustum(p, f);
        }
        if (culled) {
            for (int i = 0; i < 4; ++i) {
                // Signal degenerate frustum
                f.origins[i] = vector3(INFINITY, INFINITY, INFINITY);
                f.directions[i] = vector3(0, 0, 0);
            }
        }
        (*outPtr) = f;
    }
}

void CalculateWorldSpaceFrusta(SimpleRayFrustum* blockFrustaWS,
                               SimpleRayFrustum* tileFrustaWS,
                               SimpleRayFrustum* blockFrustaES,
                               SimpleRayFrustum* tileFrustaES,
                               matrix4x4 eyeToWorldMatrix,
                               FourPlanes cullPlanes,
                               uint32_t blockCount,
                               uint32_t tileCount,
                               cudaStream_t stream) {
    static_assert((TILE_SIZE % 32 == 0), "TILE_SIZE must be a multiple of 32, the CUDA warp size");
    size_t combinedBlockAndTileCount = tileCount + blockCount;
    KernelDim dim = KernelDim(combinedBlockAndTileCount, CUDA_GROUP_SIZE);
    CalculateWorldSpaceFrustaKernel<<<dim.grid, dim.block, 0, stream>>>(
        blockFrustaWS, tileFrustaWS, blockFrustaES, tileFrustaES, eyeToWorldMatrix, cullPlanes, blockCount, tileCount);
}

} // namespace hvvr
