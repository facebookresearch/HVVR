/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "blockcull.h"
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
#include "vector_math.h"
#include "warp_ops.h"


namespace hvvr {

// TODO(mmara): Handle beyond single point of origin rays
void ComputeEyeSpaceFrusta(const GPUBuffer<PrecomputedDirectionSample>& dirSamples,
                           GPUBuffer<SimpleRayFrustum>& tileFrusta,
                           GPUBuffer<SimpleRayFrustum>& blockFrusta) {
    DynamicArray<PrecomputedDirectionSample> samples = makeDynamicArray(dirSamples);
    DynamicArray<SimpleRayFrustum> tFrusta = makeDynamicArray(tileFrusta);
    DynamicArray<SimpleRayFrustum> bFrusta = makeDynamicArray(blockFrusta);

    auto generateFrusta = [](DynamicArray<SimpleRayFrustum>& frusta, unsigned int frustaSampleCount,
                             const DynamicArray<PrecomputedDirectionSample>& samples, float slopFactor,
                             int numOrientationsToTry) {
        auto toDir = [](const matrix3x3& rot, float u, float v) {
            return rot * normalize(vector3(u, v, 1.0f));
        };
        for (int i = 0; i < frusta.size(); ++i) {
            int sBegin = i * frustaSampleCount;
            int sEnd = min((int)((i + 1) * frustaSampleCount), (int)samples.size());
            vector3 dominantDirection(0.0f);
            for (int s = sBegin; s < sEnd; ++s) {
                // printf("samples[%d].center : %f, %f, %f\n", s, samples[s].center.x, samples[s].center.y,
                // samples[s].center.z);
                dominantDirection += samples[s].center;
            }
            dominantDirection = normalize(dominantDirection);
            // printf("Dominant Direction %d: %f, %f, %f\n", i, dominantDirection.x, dominantDirection.y,
            // dominantDirection.z);

            // Try several different orientations for the plane, pick the one that 
            // gives the smallest bounding box in uv space
            matrix3x3 rot(matrix3x3::rotationFromZAxis(dominantDirection));
            float bestUVArea = INFINITY;
            matrix3x3 bestRot;
            vector2 bestMinUV = vector2(INFINITY);
            vector2 bestMaxUV = vector2(-INFINITY);
            for (int o = 0; o < numOrientationsToTry; ++o) {
                matrix3x3 currRot = matrix3x3::axisAngle(vector3(0, 0, 1), (Pi * o / float(numOrientationsToTry))) * rot;
                matrix3x3 invCurrRot = invert(currRot);
                vector2 minUV = vector2(INFINITY);
                vector2 maxUV = vector2(-INFINITY);
                for (int s = sBegin; s < sEnd; ++s) {
                    vector3 v = invCurrRot * samples[s].center;
                    vector2 uv = vector2(v.x / v.z, v.y / v.z);

                    // TODO: check math here
                    v = invCurrRot * samples[s].d1;
                    float uvRadius = length(uv - vector2(v.x / v.z, v.y / v.z));
                    v = invCurrRot * samples[s].d2;
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

#if 0
            // Make sure all samples points are within the frustum...
            RayPacketFrustum3D checker(f);
            for (int s = sBegin; s < sEnd; ++s) {
                vector4 v = toVec(samples[s].center);
                if (!checker.testPoint(v)) {
                    printf("TROUBLE!\n");
                }
            }
#endif

            for (int o = 0; o < 4; ++o) {
                printf("f[%d].directions[%d]: %f, %f, %f\n", i, o, f.directions[o].x, f.directions[o].y,
                       f.directions[o].z);
            }
            printf("f[%d].bestUVArea: %f\n", i, bestUVArea);

            frusta[i] = f;
        }
    };
    generateFrusta(tFrusta, TILE_SIZE, samples, 4.0f, 64);
    generateFrusta(bFrusta, BLOCK_SIZE, samples, 4.0f, 64);

    tileFrusta = makeGPUBuffer(tFrusta);
    blockFrusta = makeGPUBuffer(bFrusta);
}

CUDA_KERNEL void CalculateSampleCullFrustaKernel(GPURayPacketFrustum* d_blockFrusta,
                                                 GPURayPacketFrustum* d_tileFrusta,
                                                 SampleInfo sampleInfo,
                                                 const uint32_t sampleCount) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tileIndex, blockIndex;
    float minLocX = CUDA_INF;
    float minLocY = CUDA_INF;
    float negMaxLocX = CUDA_INF;
    float negMaxLocY = CUDA_INF;
    if (index < sampleCount) {
        tileIndex = index / TILE_SIZE;
        blockIndex = index / BLOCK_SIZE;
        // TODO: see if we can cut down on work here
        UnpackedSample s = GetFullSample(index, sampleInfo);
        vector2 location = s.center;
        float radius = sampleInfo.extents[index].majorAxisLength;
        if (location.x != CUDA_INF) {
            minLocX = location.x - radius;
            minLocY = location.y - radius;
            negMaxLocX = -(location.x + radius);
            negMaxLocY = -(location.y + radius);
        }
    }
    // Do warp reduction
    auto minOp = [](float a, float b) -> float { return min(a, b); };
    minLocX = warpReduce(minLocX, minOp);
    minLocY = warpReduce(minLocY, minOp);
    negMaxLocX = warpReduce(negMaxLocX, minOp);
    negMaxLocY = warpReduce(negMaxLocY, minOp);
    int lane = threadIdx.x % WARP_SIZE;
    // All min values are in lane 0, go ahead and atomic min the results
    if (lane == 0 && (index < sampleCount)) {
        // No native float atomics, so we need to encode to handle our floats
        atomicMin((uint32_t*)(&d_tileFrusta[tileIndex].xMin), FloatFlipF(minLocX));
        atomicMin((uint32_t*)(&d_tileFrusta[tileIndex].yMin), FloatFlipF(minLocY));
        atomicMin((uint32_t*)(&d_tileFrusta[tileIndex].xNegMax), FloatFlipF(negMaxLocX));
        atomicMin((uint32_t*)(&d_tileFrusta[tileIndex].yNegMax), FloatFlipF(negMaxLocY));

        atomicMin((uint32_t*)(&d_blockFrusta[blockIndex].xMin), FloatFlipF(minLocX));
        atomicMin((uint32_t*)(&d_blockFrusta[blockIndex].yMin), FloatFlipF(minLocY));
        atomicMin((uint32_t*)(&d_blockFrusta[blockIndex].xNegMax), FloatFlipF(negMaxLocX));
        atomicMin((uint32_t*)(&d_blockFrusta[blockIndex].yNegMax), FloatFlipF(negMaxLocY));
    }
}
CUDA_KERNEL void DecodeSampleCullFrustaKernel(GPURayPacketFrustum* d_blockFrusta,
                                              uint32_t blockCount,
                                              GPURayPacketFrustum* d_tileFrusta,
                                              uint32_t tileCount) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < blockCount + tileCount) {
        uint32_t* ptr;
        if (index < blockCount) {
            ptr = (uint32_t*)(&d_blockFrusta[index]);
        } else {
            ptr = (uint32_t*)(&d_tileFrusta[index - blockCount]);
        }
        for (int i = 0; i < 4; ++i) {
            ptr[i] = IFloatFlip(ptr[i]);
        }
    }
}

CUDA_KERNEL void ResetCullFrustaKernel(GPURayPacketFrustum* d_blockFrusta,
                                       uint32_t blockCount,
                                       GPURayPacketFrustum* d_tileFrusta,
                                       uint32_t tileCount) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < blockCount + tileCount) {
        GPURayPacketFrustum* ptr;
        if (index < blockCount) {
            ptr = &d_blockFrusta[index];
        } else {
            ptr = &d_tileFrusta[index - blockCount];
        }
        // Set to -INFINITY, as its the FlipFloat encoding of +INFINITY
        ptr[0].xMin = -CUDA_INF;
        ptr[0].xNegMax = -CUDA_INF;
        ptr[0].yMin = -CUDA_INF;
        ptr[0].yNegMax = -CUDA_INF;
    }
}

void ResetCullFrusta(GPURayPacketFrustum* d_blockFrusta,
                     GPURayPacketFrustum* d_tileFrusta,
                     const uint32_t tileCount,
                     const uint32_t blockCount,
                     cudaStream_t stream) {
    {
        size_t combinedBlockAndTileCount = tileCount + blockCount;
        KernelDim dim = KernelDim(combinedBlockAndTileCount, CUDA_BLOCK_SIZE);
        ResetCullFrustaKernel<<<dim.grid, dim.block, 0, stream>>>(d_blockFrusta, blockCount, d_tileFrusta, tileCount);
    }
}

void CalculateSampleCullFrusta(GPURayPacketFrustum* d_blockFrusta,
                               GPURayPacketFrustum* d_tileFrusta,
                               SampleInfo sampleInfo,
                               const uint32_t sampleCount,
                               const uint32_t tileCount,
                               const uint32_t blockCount,
                               cudaStream_t stream) {
    static_assert((TILE_SIZE % 32 == 0), "TILE_SIZE must be a multiple of 32, the CUDA warp size");
    {
        KernelDim dim = KernelDim(sampleCount, CUDA_BLOCK_SIZE);
        CalculateSampleCullFrustaKernel<<<dim.grid, dim.block, 0, stream>>>(d_blockFrusta, d_tileFrusta, sampleInfo,
                                                                            sampleCount);
    }
    {
        size_t combinedBlockAndTileCount = tileCount + blockCount;
        KernelDim dim = KernelDim(combinedBlockAndTileCount, CUDA_BLOCK_SIZE);
        DecodeSampleCullFrustaKernel<<<dim.grid, dim.block, 0, stream>>>(d_blockFrusta, blockCount, d_tileFrusta,
                                                                         tileCount);
    }
}

CUDA_KERNEL void CalculateWorldSpaceFrustaKernel(SimpleRayFrustum* blockFrustaWS,
                                                 SimpleRayFrustum* tileFrustaWS,
                                                 SimpleRayFrustum* blockFrustaES,
                                                 SimpleRayFrustum* tileFrustaES,
                                                 matrix4x4 eyeToWorldMatrix,
                                                 uint32_t blockCount,
                                                 uint32_t tileCount) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < blockCount + tileCount) {
        SimpleRayFrustum* inPtr;
        SimpleRayFrustum* outPtr;
        if (index < blockCount) {
            inPtr = (&blockFrustaES[index]);
            outPtr = (&blockFrustaWS[index]);
        } else {
            inPtr = (&tileFrustaES[index - blockCount]);
            outPtr = (&tileFrustaWS[index - blockCount]);
        }
        for (int i = 0; i < 4; ++i) {
            vector4 origin =
                eyeToWorldMatrix * vector4((*inPtr).origins[i].x, (*inPtr).origins[i].y, (*inPtr).origins[i].z, 1.0f);
            (*outPtr).origins[i].x = origin.x;
            (*outPtr).origins[i].y = origin.y;
            (*outPtr).origins[i].z = origin.z;
            // TODO(mmara): use inverse transpose to handle non-uniform scale?
            vector4 direction = eyeToWorldMatrix * vector4((*inPtr).directions[i].x, (*inPtr).directions[i].y,
                                                           (*inPtr).directions[i].z, 0.0f);
            (*outPtr).directions[i].x = direction.x;
            (*outPtr).directions[i].y = direction.y;
            (*outPtr).directions[i].z = direction.z;
        }
    }
}

void CalculateWorldSpaceFrusta(SimpleRayFrustum* blockFrustaWS,
                               SimpleRayFrustum* tileFrustaWS,
                               SimpleRayFrustum* blockFrustaES,
                               SimpleRayFrustum* tileFrustaES,
                               matrix4x4 eyeToWorldMatrix,
                               uint32_t blockCount,
                               uint32_t tileCount,
                               cudaStream_t stream) {
    static_assert((TILE_SIZE % 32 == 0), "TILE_SIZE must be a multiple of 32, the CUDA warp size");
    size_t combinedBlockAndTileCount = tileCount + blockCount;
    KernelDim dim = KernelDim(combinedBlockAndTileCount, CUDA_BLOCK_SIZE);
    CalculateWorldSpaceFrustaKernel<<<dim.grid, dim.block, 0, stream>>>(
        blockFrustaWS, tileFrustaWS, blockFrustaES, tileFrustaES, eyeToWorldMatrix, blockCount, tileCount);
}

} // namespace hvvr
