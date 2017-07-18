/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "cuda_decl.h"
#include "gbuffer.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "kernel_constants.h"
#include "prim_tests.h"
#include "sort.h"
#include "tile_data.h"
#include "warp_ops.h"

#define PROFILE_INTERSECT 0


namespace hvvr {

enum : uint32_t { badTriIndex = ~uint32_t(0) };

// returns zero if, for the entire warp, all lanes hit 1 triangle or 0 triangles - the fast path, a single gbuffer entry
// returns the number of unique triangles for each thread
// note that 1 of the unique triangle count will be badTriIndex if one or more subsamples don't hit a triangle
template <uint32_t MSAARate, uint32_t BlockSize>
CUDA_DEVICE int EmitGBuffer(int laneIndex, const uint32_t* sampleTriIndex, RaycasterGBufferSubsample* gBufferWarp) {
    // fast path if all lanes hit 1 triangle
    bool oneTri = true;
    uint32_t triIndex0 = sampleTriIndex[0];
    for (int i = 1; i < MSAARate; i++) {
        if (triIndex0 != sampleTriIndex[i]) {
            oneTri = false;
        }
    }
    bool oneTriPerLane = (ballot(oneTri) == laneGetMaskAll());

    if (oneTriPerLane) {
        // for all threads in the warp, only a single triangle is hit
        // not necessarily the same triangle across threads, though
        RaycasterGBufferSubsample gBufferSample;
        gBufferSample.triIndex = sampleTriIndex[0];
        gBufferSample.sampleMask = (1 << MSAARate) - 1;

        gBufferWarp[laneIndex] = gBufferSample;

        return 0;
    } else {
        // group hits on the same triangle together, and misses (badTriIndex) at the end
        uint32_t sortKeys[MSAARate];
        // max triIndex is 134217728 - 1 ... seems a reasonable cap for now (triangle data would exceed the VRAM of most
        // GPUs) max samples is 32
        enum { sortKeySampleBits = 5 }; // this sort and gbuffer compression (sampleMask) assume <= 32 samples
        enum { sortKeySampleMask = (1 << sortKeySampleBits) - 1 };
        enum { sortKeyIndexBits = 32 - sortKeySampleBits };
        static_assert(MSAARate <= (1 << sortKeySampleBits),
                      "triangle sample sort and gbuffer compression assume a maximum MSAARate");
        for (int i = 0; i < MSAARate; i++) {
            sortKeys[i] = (sampleTriIndex[i] << sortKeySampleBits) | i;
        }
        sortBitonic(sortKeys, MSAARate);

        // scan and reduce down to pairs of (triIndex, sampleMask)
        RaycasterGBufferSubsample gBufferSample;
        gBufferSample.triIndex = int32_t(sortKeys[0]) >> sortKeySampleBits; // shift with sign fill
        gBufferSample.sampleMask = 1 << (sortKeys[0] & sortKeySampleMask);
        int compTriCount = 1;
        for (int i = 1; i < MSAARate; i++) {
            uint32_t triIndex = int32_t(sortKeys[i]) >> sortKeySampleBits;
            uint32_t triMask = 1 << (sortKeys[i] & sortKeySampleMask);

            if (gBufferSample.triIndex == triIndex) {
                gBufferSample.sampleMask |= triMask;
            } else {
                // we found a new triangle ID, emit the existing compressed sample pair and start a new one
                gBufferWarp[(compTriCount - 1) * WARP_SIZE + laneIndex] = gBufferSample;

                gBufferSample.triIndex = triIndex;
                gBufferSample.sampleMask = triMask;
                compTriCount++;
            }
        }
        gBufferWarp[(compTriCount - 1) * WARP_SIZE + laneIndex] = gBufferSample;

        return compTriCount;
    }
}

template <uint32_t MSAARate, uint32_t BlockSize>
struct TriCache {
    enum { maxSize = BlockSize };

    // TODO(anankervis): don't waste space by unioning the DoF and non-DoF variants (getting the bigger size of the two)
    union {
        IntersectTriangleTile data[maxSize];
        IntersectTriangleTileDoF dataDoF[maxSize];
    };
    uint32_t index[maxSize];

    enum { warpCount = (BlockSize + WARP_SIZE - 1) / WARP_SIZE };
    int sizeWarp[warpCount]; // temporary used to hold the number of surviving triangles per warp

    SimpleRayFrustum frustumRays;
    FrustumPlanes frustumPlanes;

    union {
        TileData tile;
        TileDataDoF tileDoF;
    };

    uint32_t sampleTriIndex[BlockSize * MSAARate];
};

template <uint32_t MSAARate, uint32_t BlockSize, bool EnableDoF>
CUDA_DEVICE void IntersectSamples(const PrecomputedTriangleIntersect* CUDA_RESTRICT trianglesIntersect,
                                  SampleInfo sampleInfo,
                                  const UnpackedDirectionalSample& sample,
                                  matrix4x4 sampleToWorld,
                                  const vector2* CUDA_RESTRICT tileSubsampleLensPos,
                                  uint32_t sampleOffset,
                                  TriCache<MSAARate, BlockSize>& triCache,
                                  int triCount,
                                  float* sampleTMax) {
    UnpackedSample sample2D = GetFullSample(sampleOffset, sampleInfo);
    matrix3x3 sampleToWorldRotation = matrix3x3(sampleToWorld);
    vector3 lensCenterToFocalCenter =
        sampleInfo.lens.focalDistance * (sampleToWorldRotation * vector3(sample2D.center.x, sample2D.center.y, 1.0f));

    for (int j = 0; j < triCount; ++j) {
        uint32_t triIndex = triCache.index[j];

        if (EnableDoF) {
            IntersectTriangleTileDoF triTileDoF = triCache.dataDoF[j];
            IntersectTriangleThreadDoF triThreadDoF(triTileDoF, lensCenterToFocalCenter);

#pragma unroll
            for (int i = 0; i < MSAARate; ++i) {
                vector2 lensUV;
                vector2 dirUV;
                GetSampleUVsDoF<MSAARate, BlockSize>(tileSubsampleLensPos, sampleInfo.frameJitter,
                                                     triCache.tileDoF.focalToLensScale, i, lensUV, dirUV);

                if (triThreadDoF.test(triTileDoF, lensCenterToFocalCenter, triCache.tileDoF.lensU,
                                      triCache.tileDoF.lensV, lensUV, dirUV, sampleTMax[i])) {
                    // ray intersected triangle and passed depth test, sampleTMax[i] has been updated
                    // 1 STS
                    triCache.sampleTriIndex[i * BlockSize + threadIdx.x] = triIndex;
                }
            }
        } else {
            IntersectTriangleTile triTile = triCache.data[j];
            IntersectTriangleThread triThread(triTile, sample.centerDir);

#pragma unroll
            for (int i = 0; i < MSAARate; ++i) {
                vector2 alpha = getSubsampleUnitOffset<MSAARate>(sampleInfo.frameJitter, i);

                if (triThread.test(triTile, alpha, sampleTMax[i])) {
                    // ray intersected triangle and passed depth test, sampleTMax[i] has been updated
                    // 1 STS
                    triCache.sampleTriIndex[i * BlockSize + threadIdx.x] = triIndex;
                }
            }
        }
    }
}

// intersect a whole tile of rays
template <uint32_t MSAARate, uint32_t BlockSize, bool EnableDoF>
CUDA_DEVICE void IntersectTile(SampleInfo sampleInfo,
                               matrix4x4 sampleToWorld,
                               matrix3x3 sampleToCamera,
                               matrix4x4 cameraToWorld,
                               const vector2* CUDA_RESTRICT tileSubsampleLensPos,
                               const unsigned* CUDA_RESTRICT triIndices,
                               const PrecomputedTriangleIntersect* CUDA_RESTRICT trianglesIntersect,
                               uint32_t sampleOffset,
                               TileTriRange triRange,
                               TriCache<MSAARate, BlockSize>& sMemTriCache,
                               float* sampleTMax) {
    // TODO: switch to full 3D sample to allow different directions per subsample
    UnpackedDirectionalSample sample =
        GetDirectionalSample3D(sampleOffset, sampleInfo, sampleToWorld, sampleToCamera, cameraToWorld);

    // TODO(anankervis): precompute this with more accurate values, and load from a per-tile buffer
    // (but watch out for the foveated path)
    if (threadIdx.x == BlockSize / 2) {
        if (EnableDoF) {
            sMemTriCache.tileDoF.load(sampleInfo, sampleToWorld, sampleOffset);
        } else {
            sMemTriCache.tile.load(sampleToWorld, sample);
        }
        // make sure there is a __syncthreads somewhere between here and first use (tri cache init, for example)
    }

    for (int i = 0; i < MSAARate; i++) {
        sampleTMax[i] = FLT_MAX;
        sMemTriCache.sampleTriIndex[i * BlockSize + threadIdx.x] = badTriIndex;
    }
    __syncthreads();

    for (uint32_t triRangeCurrent = triRange.start; triRangeCurrent < triRange.end;
         triRangeCurrent += sMemTriCache.maxSize) {
        // each thread cooperates to populate the shared mem triangle cache
        bool outputTri = false;
        uint32_t triIndex = badTriIndex;
        IntersectTriangleTile triTile;
        IntersectTriangleTileDoF triTileDoF;
        if (threadIdx.x < sMemTriCache.maxSize) {
            uint32_t triIndirectIndex = triRangeCurrent + threadIdx.x;
            if (triIndirectIndex < triRange.end) {
                triIndex = triIndices[triIndirectIndex];
                const PrecomputedTriangleIntersect& triIntersect = trianglesIntersect[triIndex];

                // test the triangle against the tile frustum's planes
                if (sMemTriCache.frustumPlanes.test(triIntersect)) {
                    if (EnableDoF) {
                        // test for backfacing and intersection before ray origin
                        IntersectResult intersectResultSetup =
                            triTileDoF.setup(triIntersect, sMemTriCache.tileDoF.lensCenter, sMemTriCache.tileDoF.lensU,
                                             sMemTriCache.tileDoF.lensV);
                        if (intersectResultSetup != intersect_all_out) {
#if FOVEATED_TRIANGLE_FRUSTA_TEST_DISABLE == 0
                            // test the tile frustum against the triangle's edges
                            // only perform this test if all rays within the tile can be guaranteed to intersect the
                            // front side of the triangle's plane
                            IntersectResult intersectResultUVW = intersect_partial;
                            if (intersectResultSetup != intersect_partial) {
                                intersectResultUVW = TestTriangleFrustaUVW(sMemTriCache.frustumRays, triIntersect);
                            }

                            if (intersectResultUVW != intersect_all_out)
#endif
                            {
                                outputTri = true;
                            }
                        }
                    } else {
                        IntersectResult intersectResultSetup =
                            triTile.setup(triIntersect, sMemTriCache.tile.rayOrigin, sMemTriCache.tile.majorDirDiff,
                                          sMemTriCache.tile.minorDirDiff);
                        if (intersectResultSetup != intersect_all_out) {
#if FOVEATED_TRIANGLE_FRUSTA_TEST_DISABLE == 0
                            IntersectResult intersectResultUVW =
                                TestTriangleFrustaUVW(sMemTriCache.frustumRays, triIntersect);

                            if (intersectResultUVW != intersect_all_out)
#endif
                            {
                                outputTri = true;
                            }
                        }
                    }
                }
            }
        }

        // output slot within the warp
        int outputMaskWarp = warpBallot(outputTri);
        int outputCountWarp = __popc(outputMaskWarp);
        int outputIndexWarp = __popc(outputMaskWarp & laneGetMaskLT());

        // number of output slots needed by each warp
        int warpIndex = threadIdx.x / WARP_SIZE;
        if (threadIdx.x % WARP_SIZE == 0) {
            sMemTriCache.sizeWarp[warpIndex] = outputCountWarp;
        }
        // number of output slots needed by the whole block
        int outputCount = __syncthreads_count(outputTri);

        // output slot across the whole block
        int outputIndex = outputIndexWarp;
        for (int n = 0; n < warpIndex; n++) {
            outputIndex += sMemTriCache.sizeWarp[n];
        }

        if (outputTri) {
            sMemTriCache.index[outputIndex] = triIndex;
            if (EnableDoF) {
                sMemTriCache.dataDoF[outputIndex] = triTileDoF;
            } else {
                sMemTriCache.data[outputIndex] = triTile;
            }
        }
        __syncthreads();

        IntersectSamples<MSAARate, BlockSize, EnableDoF>(trianglesIntersect, sampleInfo, sample, sampleToWorld,
                                                         tileSubsampleLensPos, sampleOffset, sMemTriCache, outputCount,
                                                         sampleTMax);

        // TODO(anankervis): for some reason, this is needed to prevent corruption when storing sampleTriIndex in sMem
        __syncthreads();
    }
}

// remember to __syncthreads() before aliasing sMem
template <uint32_t MSAARate, uint32_t BlockSize>
union IntersectTileSharedMem {
    TriCache<MSAARate, BlockSize> triCache;

    IntersectTileSharedMem(){};
};

template <uint32_t MSAARate, uint32_t BlockSize, bool EnableDoF>
CUDA_KERNEL void IntersectKernel(RaycasterGBufferSubsample* gBuffer,
                                 SampleInfo sampleInfo,
                                 matrix4x4 sampleToWorld,
                                 matrix3x3 sampleToCamera,
                                 matrix4x4 cameraToWorld,
                                 const vector2* CUDA_RESTRICT tileSubsampleLensPos,
                                 const uint32_t* CUDA_RESTRICT tileIndexRemapOccupied,
                                 const TileTriRange* CUDA_RESTRICT tileTriRanges,
                                 const uint32_t* CUDA_RESTRICT triIndices,
                                 const SimpleRayFrustum* CUDA_RESTRICT tileFrusta,
                                 const PrecomputedTriangleIntersect* CUDA_RESTRICT trianglesIntersect) {
    static_assert(TILE_SIZE == BlockSize, "IntersectKernel assumes TILE_SIZE == BlockSize");

    uint32_t rayInTileIndex = threadIdx.x;
    uint32_t compactedTileIndex = blockIdx.x;
    uint32_t tileIndex = tileIndexRemapOccupied[compactedTileIndex];
    uint32_t sampleOffset = tileIndex * TILE_SIZE + rayInTileIndex;

    // grab the range of indirect triangle indices this block is expected to process
    TileTriRange triRange = tileTriRanges[compactedTileIndex];

    __shared__ IntersectTileSharedMem<MSAARate, BlockSize> sMem;
    if (threadIdx.x == 0) {
        SimpleRayFrustum frustumRays = tileFrusta[tileIndex];
        sMem.triCache.frustumRays = frustumRays;
        sMem.triCache.frustumPlanes = FrustumPlanes(frustumRays);
        // make sure there is a __syncthreads somewhere between here and first use (tri cache init, for example)
    }

    float sampleTMax[MSAARate];
    IntersectTile<MSAARate, BlockSize, EnableDoF>(sampleInfo, sampleToWorld, sampleToCamera, cameraToWorld,
                                                  tileSubsampleLensPos, triIndices, trianglesIntersect, sampleOffset,
                                                  triRange, sMem.triCache, sampleTMax);

    uint32_t sampleTriIndex[MSAARate];
    for (int i = 0; i < MSAARate; i++) {
        sampleTriIndex[i] = sMem.triCache.sampleTriIndex[i * BlockSize + threadIdx.x];
    }

    __syncthreads(); // before aliasing sMem

    // GBuffer texels are organized so that each subsample is a warp stride away from
    // the previous subsample for a single sample, so that warps can coalesce memory reads
    uint32_t warpIndex = sampleOffset / WARP_SIZE;
    uint32_t warpOffset = warpIndex * WARP_SIZE * MSAARate;
    EmitGBuffer<MSAARate, BlockSize>(laneGetIndex(), sampleTriIndex, gBuffer + warpOffset);
}

#if PROFILE_INTERSECT
CUDA_KERNEL void GenerateHeat() {
    int* somewhere = (int*)0xFEDCBA9876543210;
    int value = 0;
    int time = clock();
    while (clock() - time < 10000000) {
        value += clock();
    }
    if (blockIdx.x == 65000) {
        *somewhere = value;
    }
}
#endif

void GPUCamera::intersect(GPUSceneState& sceneState, const SampleInfo& sampleInfo) {
    Camera_StreamedData& streamedData = streamed[streamedIndexGPU];

    uint32_t occupiedTileCount = streamedData.tileCountOccupied;

#if PROFILE_INTERSECT
    static uint64_t frameIndex = 0;
    enum { profileFrameSkip = 64 };
    static cudaEvent_t start = nullptr;
    static cudaEvent_t stop = nullptr;
    static float minTimeMs = FLT_MAX;
    if (!start) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    if (frameIndex % profileFrameSkip == 0) {
        cudaEventRecord(start, stream);
    }
#endif

    KernelDim dimIntersect(occupiedTileCount * TILE_SIZE, TILE_SIZE);
    if (sampleInfo.lens.radius > 0.0f) {
        // Enable depth of field
        IntersectKernel<COLOR_MODE_MSAA_RATE, TILE_SIZE, true><<<dimIntersect.grid, dimIntersect.block, 0, stream>>>(
            d_gBuffer, sampleInfo, cameraToWorld * matrix4x4(sampleToCamera), sampleToCamera, cameraToWorld,
            d_tileSubsampleLensPos, local.tileIndexRemapOccupied, local.tileTriRanges, streamedData.triIndices,
            local.tileFrusta3D, sceneState.trianglesIntersect);
    } else {
        // No depth of field, assume all rays have the same origin
        IntersectKernel<COLOR_MODE_MSAA_RATE, TILE_SIZE, false><<<dimIntersect.grid, dimIntersect.block, 0, stream>>>(
            d_gBuffer, sampleInfo, cameraToWorld * matrix4x4(sampleToCamera), sampleToCamera, cameraToWorld,
            d_tileSubsampleLensPos, local.tileIndexRemapOccupied, local.tileTriRanges, streamedData.triIndices,
            local.tileFrusta3D, sceneState.trianglesIntersect);
    }

#if PROFILE_INTERSECT
    if (frameIndex % profileFrameSkip == 0) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float timeMs = 0.0f;
        cudaEventElapsedTime(&timeMs, start, stop);
        minTimeMs = min(minTimeMs, timeMs);
        printf("intersect min: %.2fms\n", minTimeMs);
    }
    frameIndex++;

    // I need more of a workload to get consistent clocks out of the GPU...
    GenerateHeat<<<1024, 32, 0, stream>>>();
#endif
}

template <uint32_t MSAARate>
CUDA_KERNEL void DumpRaysKernel(SimpleRay* rayBuffer,
                                SampleInfo sampleInfo,
                                matrix4x4 sampleToWorld,
                                matrix3x3 sampleToCamera,
                                matrix4x4 cameraToWorld,
                                const vector2* CUDA_RESTRICT tileSubsampleLensPos,
                                int sampleCount) {
    uint32_t sampleOffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleOffset < sampleCount) {
        UnpackedDirectionalSample sample3D =
            GetDirectionalSample3D(sampleOffset, sampleInfo, sampleToWorld, sampleToCamera, cameraToWorld);

        for (int i = 0; i < MSAARate; ++i) {
            vector2 alpha = getSubsampleUnitOffset<MSAARate>(sampleInfo.frameJitter, i);
            vector3 dir = sample3D.centerDir + sample3D.majorDirDiff * alpha.x + sample3D.minorDirDiff * alpha.y;
            vector3 pos = vector3(cameraToWorld * vector4(0, 0, 0, 1));
            if (sampleInfo.lens.radius > 0.0f) {
                // TODO: implement
                /*SampleDoF sampleDoF = GetSampleDoF<MSAARate, TILE_SIZE>(sampleOffset, i, sampleInfo, sampleToCamera,
                                                                        cameraToWorld, tileSubsampleLensPos);
                pos = sampleDoF.pos;
                dir = sampleDoF.dir;*/
            }
            uint32_t index = sampleOffset * MSAARate + i;
            rayBuffer[index].direction.x = dir.x;
            rayBuffer[index].direction.y = dir.y;
            rayBuffer[index].direction.z = dir.z;
            rayBuffer[index].origin.x = pos.x;
            rayBuffer[index].origin.y = pos.y;
            rayBuffer[index].origin.z = pos.z;
        }
    }
}

void GPUCamera::dumpRays(std::vector<SimpleRay>& rays) {
    uint32_t rayCount = COLOR_MODE_MSAA_RATE * sampleCount;
    GPUBuffer<SimpleRay> d_rays(rayCount);
    rays.resize(rayCount);

    SampleInfo sampleInfo(*this);
    uint32_t tileCount = (sampleCount + TILE_SIZE - 1) / TILE_SIZE;

    KernelDim dim(tileCount * TILE_SIZE, TILE_SIZE);
    DumpRaysKernel<COLOR_MODE_MSAA_RATE>
        <<<dim.grid, dim.block, 0, stream>>>(d_rays, sampleInfo, cameraToWorld * matrix4x4(sampleToCamera),
                                             sampleToCamera, cameraToWorld, d_tileSubsampleLensPos.data(), sampleCount);

    d_rays.readback(rays.data());
}

} // namespace hvvr
