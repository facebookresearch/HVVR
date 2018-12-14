/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#define TRAVERSAL_IMP 1 // pull in any extra inline functions needed for traversal
#include "traversal.h"

#include "avx.h"
#include "camera.h"
#include "constants_math.h"
#include "gpu_camera.h"
#include "magic_constants.h"
#include "raycaster.h"
#include "thread_pool.h"
#include "timer.h"

#include <stdio.h>

// If enabled, compares the chosen traversal implementation (see TRAVERSAL_MODE) against TRAVERSAL_REF
// and prints stats.
#define VERIFY_TRAVERSAL 0
#if VERIFY_TRAVERSAL
# include "traverse_ref.h"
# include <atomic>
static std::atomic<int> impBlockStackSum;
static std::atomic<int> refBlockStackSum;
static std::atomic<int> impTileTriSum;
static std::atomic<int> refTileTriSum;
#endif

#define DEBUG_STATS 0
#define TIME_BLOCK_CULL 0

#pragma warning(disable : 4505) // unreferenced local function

namespace hvvr {

// This could very easily be drastically simplified. But it is debugging code...
// http://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
static double angleOnSphere(vector3 A, vector3 B, vector3 C) {
    vector3 AcrossB = cross(A, B);
    vector3 CcrossB = cross(C, B);
    return acosf(dot(AcrossB, CcrossB) / (length(AcrossB) * length(CcrossB)));
}

// http://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
static double sphericalTriangleArea(vector3 A, vector3 B, vector3 C) {
    return angleOnSphere(A, B, C) + angleOnSphere(B, C, A) + angleOnSphere(C, A, B) - M_PI;
}

// Only works for frusta with center of projection
static double solidAngle(Frustum frustum) {
    vector3 v[4] = {normalize(frustum.pointDir[0]), normalize(frustum.pointDir[1]), normalize(frustum.pointDir[2]),
                    normalize(frustum.pointDir[3])};

    return sphericalTriangleArea(v[0], v[1], v[2]) + sphericalTriangleArea(v[0], v[2], v[3]);
}

static vector4 minMaxMeanMedian(std::vector<double>& vec, size_t& validCount) {
    std::vector<double> cleanVec;
    for (int i = 0; i < vec.size(); ++i) {
        if (!isnan(vec[i])) {
            cleanVec.push_back(vec[i]);
        }
    }
    validCount = cleanVec.size();
    // printf("valid/total = %zd/%zd = %f\n", cleanVec.size(), vec.size(), cleanVec.size() / (double)vec.size());
    double minV = INFINITY;
    double maxV = -INFINITY;
    double sumV = 0.0f;
    std::sort(cleanVec.begin(), cleanVec.end());
    for (auto d : cleanVec) {
        sumV += d;
        minV = min(minV, d);
        maxV = max(maxV, d);
    }
    return vector4(float(minV), float(maxV), float(sumV) / cleanVec.size(), float(cleanVec[cleanVec.size() / 2]));
};

struct TaskData {
    uint32_t triIndexCount;
    std::vector<unsigned> tileIndexRemapOccupied;
    std::vector<unsigned> tileIndexRemapEmpty;
    std::vector<TileTriRange> tileTriRanges;
    ArrayView<unsigned> triIndices;
    void reset(ArrayView<unsigned> fullTriIndices, size_t bufferStart, size_t bufferEnd) {
        triIndexCount = 0;
        triIndices = ArrayView<unsigned>(fullTriIndices.data() + bufferStart, bufferEnd - bufferStart);
        tileIndexRemapOccupied.clear();
        tileIndexRemapEmpty.clear();
        tileTriRanges.clear();
    }
};

static void cullThread(const RayHierarchy& rayHierarchy,
                       uint32_t startBlock,
                       uint32_t endBlock,
                       const BVHNode* nodes,
                       TaskData* perThread) {
#if DEBUG_STATS
    auto startTime = (double)__rdtsc();
#endif

    if (startBlock == endBlock) {
        return;
    }

    perThread->triIndexCount = 0;

    BlockFrame blockFrame;
    for (uint32_t b = startBlock; b < endBlock; ++b) {
        const Frustum& blockFrustum = rayHierarchy.blockFrusta[b];
        uint32_t stackSize = traverseBlocks(blockFrame, nodes, blockFrustum);
#if VERIFY_TRAVERSAL
        traverse::ref::BlockFrame refBlockFrame;
        traverse::ref::Frustum refBlockFrustum(blockFrustum.pointOrigin, blockFrustum.pointDir);
        uint32_t refStackSize = traverse::ref::traverseBlocks(refBlockFrame, nodes, refBlockFrustum);

        impBlockStackSum += stackSize;
        refBlockStackSum += refStackSize;

        traverse::ref::TileFrame refTileFrameInit(refBlockFrame, refStackSize);
#endif

        if (!stackSize) { // we hit nothing?
            for (unsigned i = 0; i < TILES_PER_BLOCK; ++i) {
                auto globalTileIndex = b * TILES_PER_BLOCK + i;
                perThread->tileIndexRemapEmpty.push_back(globalTileIndex);
            }
            continue;
        }

#if TRAVERSAL_MODE == TRAVERSAL_REF
        TileFrame tileFrameInit(blockFrame, stackSize);
#elif TRAVERSAL_MODE == TRAVERSAL_AVX
        blockFrame.sort(stackSize);
#else
# error unknown traversal mode
#endif

        auto i = TILES_PER_BLOCK;
        do {
            auto tileIndex = (TILES_PER_BLOCK - i);
            auto globalTileIndex = b * TILES_PER_BLOCK + tileIndex;

#if TRAVERSAL_MODE == TRAVERSAL_REF
            TileFrame tileFrame(tileFrameInit, stackSize);
#elif TRAVERSAL_MODE == TRAVERSAL_AVX
            TileFrame tileFrame;
            for (uint32_t slot = 0; slot != stackSize; ++slot)
                store(&(tileFrame.stack + slot)->tMin, load_m128(&(blockFrame.sortedStack + slot)->tMin));
#else
# error unknown traversal mode
#endif
            uint32_t* triIndices = perThread->triIndices.data() + perThread->triIndexCount;
            uint32_t maxTriCount = uint32_t(perThread->triIndices.size()) - perThread->triIndexCount;

            const Frustum& tileFrustum = rayHierarchy.tileFrusta[globalTileIndex];
            uint32_t outputTriCount = traverseTiles(triIndices, maxTriCount, tileFrame, stackSize, tileFrustum);
#if VERIFY_TRAVERSAL
            traverse::ref::TileFrame refTileFrame(refTileFrameInit, stackSize);
            traverse::ref::Frustum refTileFrustum(tileFrustum.pointOrigin, tileFrustum.pointDir);
            uint32_t refOutputTriCount = traverse::ref::traverseTiles(
                triIndices, maxTriCount, refTileFrame, stackSize, refTileFrustum);

            impTileTriSum += outputTriCount;
            refTileTriSum += refOutputTriCount;

            // we just overwrote the triangle buffer with our verification run...
            // so we need to use the verification triangle count to avoid artifacts
            outputTriCount = refOutputTriCount;
#endif

            if (outputTriCount) {
                TileTriRange triRange;
                triRange.start = perThread->triIndexCount;
                triRange.end = triRange.start + outputTriCount;

                perThread->tileTriRanges.push_back(triRange);
                perThread->tileIndexRemapOccupied.push_back(globalTileIndex);
                perThread->triIndexCount += outputTriCount;
            } else {
                perThread->tileIndexRemapEmpty.push_back(globalTileIndex);
            }
        } while (--i);
    }

#if DEBUG_STATS
    // double deltaTimeMs = ((double)__rdtsc() - startTime) * whunt::gRcpCPUFrequency * 1000.0;
    std::vector<double> blockFrustaAngle(endBlock - startBlock);
    std::vector<double> tileFrustaAngle((endBlock - startBlock) * TILES_PER_BLOCK);
    for (size_t b = startBlock; b < endBlock; ++b) {
        blockFrustaAngle[b - startBlock] = solidAngle(rayHierarchy.blockFrusta[b]);
        for (size_t t = 0; t < TILES_PER_BLOCK; ++t) {
            tileFrustaAngle[t] = solidAngle(rayHierarchy.tileFrusta[b * TILES_PER_BLOCK + t]);
        }
    }
    size_t validBlocks, validTiles;
    vector4 m4Block = minMaxMeanMedian(blockFrustaAngle, validBlocks);
    vector4 m4Tile = minMaxMeanMedian(tileFrustaAngle, validTiles);

    // printf("---- Block cull [%u,%u) solid angle:  time %f, %u triangles,  %g percent coverage\n", startBlock,
    // endBlock, deltaTimeMs, currentTriIdx, 100.0*m4X.z*validBlocks / (4 * M_PI));
    printf("%u, %u, %g, %g\n", startBlock, perThread->triIndexCount, 100.0 * m4Block.z * validBlocks / (4 * M_PI),
           100.0 * m4Tile.z * validTiles / (4 * M_PI));
#endif
}

uint64_t Raycaster::buildTileTriangleLists(const RayHierarchy& rayHierarchy, Camera_StreamedData* streamed) {
    const BVHNode* nodes = _nodes.data();
    ArrayView<uint32_t> triIndices(streamed->triIndices.dataHost(), streamed->triIndices.size());

#if DEBUG_STATS
    std::vector<double> blockFrustaAngle(rayHierarchy.blockFrusta.size());
    for (int i = 0; i < rayHierarchy.blockFrusta.size(); ++i) {
        blockFrustaAngle[i] = solidAngle(rayHierarchy.blockFrusta[i]);
    }
    size_t validBlocks;
    vector4 m4X = minMaxMeanMedian(blockFrustaAngle, validBlocks);
    printf("Block: Min,Max,Mean,Median: %g, %g, %g, %g\n", m4X.x, m4X.y, m4X.z, m4X.w);
    printf("Percent of sphere covered by block frusta: %g\n", 100.0 * m4X.z * validBlocks / (4 * M_PI));

    std::vector<double> tileFrustaAngle(rayHierarchy.tileFrusta.size());
    for (int i = 0; i < rayHierarchy.tileFrusta.size(); ++i) {
        // Convert to square degrees from steradians
        tileFrustaAngle[i] = solidAngle(rayHierarchy.tileFrusta[i]);
    }
    size_t validTiles;
    m4X = minMaxMeanMedian(tileFrustaAngle, validTiles);
    printf("Tile: Min,Max,Mean,Median: %g, %g, %g, %g\n", m4X.x, m4X.y, m4X.z, m4X.w);
    printf("Percent of sphere covered by tile frusta: %g\n", 100.0 * m4X.z * validTiles / (4 * M_PI));
#endif

#if DEBUG_STATS || TIME_BLOCK_CULL
    Timer timer;
#endif

#if VERIFY_TRAVERSAL
    impBlockStackSum = 0;
    refBlockStackSum = 0;
    impTileTriSum = 0;
    refTileTriSum = 0;
#endif

    enum { maxTasks = 4096 };
    // workload per task
    // 1 is too low, and incurs overhead from switching tasks too frequently inside the thread pool
    // 2 seems the fastest (though this could vary depending on the scene and sample distribution)
    // 3+ seems to become less efficient due to workload balancing
    enum { blocksPerThread = 2 };
    uint32_t blockCount = uint32_t(rayHierarchy.blockFrusta.size());
    uint32_t numTasks = (blockCount + blocksPerThread - 1) / blocksPerThread;
    assert(numTasks <= maxTasks);
    numTasks = min<uint32_t>(maxTasks, numTasks);
    uint32_t triSpacePerThread = uint32_t(triIndices.size() / numTasks);
    assert(triSpacePerThread * numTasks <= triIndices.size());

    std::future<void> taskResults[maxTasks];
    static TaskData taskData[maxTasks];
    for (uint32_t i = 0; i < numTasks; ++i) {
        uint32_t startTriIndex = i * triSpacePerThread;
        uint32_t endTriIndex = startTriIndex + triSpacePerThread;
        taskData[i].reset(triIndices, startTriIndex, endTriIndex);

        uint32_t startBlock = min(blockCount, i * blocksPerThread);
        uint32_t endBlock = min(blockCount, (i + 1) * blocksPerThread);
        if (i == numTasks - 1)
            assert(endBlock == blockCount);

        taskResults[i] = _threadPool->addTask(cullThread, rayHierarchy, startBlock, endBlock, nodes, &taskData[i]);
    }

    uint64_t triIndexCount = 0;
    uint32_t maxTaskTriCount = 0;
    uint32_t tileTriOffsetsStreamed = 0;
    uint32_t* streamTileIndexRemapEmpty = streamed->tileIndexRemapEmpty.dataHost();
    uint32_t* streamTileIndexRemapOccupied = streamed->tileIndexRemapOccupied.dataHost();
    TileTriRange* streamTileTriRanges = streamed->tileTriRanges.dataHost();
    for (uint32_t taskIndex = 0; taskIndex < numTasks; taskIndex++) {
        taskResults[taskIndex].get();
        const TaskData& task = taskData[taskIndex];

        for (auto emptyTileIndex : task.tileIndexRemapEmpty) {
            streamTileIndexRemapEmpty[streamed->tileCountEmpty++] = emptyTileIndex;
        }

        for (auto occupiedTileIndex : task.tileIndexRemapOccupied) {
            streamTileIndexRemapOccupied[streamed->tileCountOccupied++] = occupiedTileIndex;
        }

        // tileTriRange within the task is relative to the task's smaller view of the buffer and needs to be offset
        uint32_t taskTriOffset = uint32_t(task.triIndices.data() - streamed->triIndices.dataHost());
        for (auto tileTriRange : task.tileTriRanges) {
            TileTriRange triRangeGlobal;
            triRangeGlobal.start = taskTriOffset + tileTriRange.start;
            triRangeGlobal.end = taskTriOffset + tileTriRange.end;
            streamTileTriRanges[tileTriOffsetsStreamed++] = triRangeGlobal;
        }

        triIndexCount += task.triIndexCount;
        maxTaskTriCount = max(maxTaskTriCount, task.triIndexCount);
    }
    (void)maxTaskTriCount;
#if DEBUG_STATS || TIME_BLOCK_CULL
    double deltaTime = timer.get();
    static double minDeltaTime = DBL_MAX;
    minDeltaTime = min(minDeltaTime, deltaTime);

    static uint64_t frameIndex = 0;
    enum { profileFrameSkip = 64 };
    if (frameIndex % profileFrameSkip == 0) {
        printf("---- Block cull time: %.2fms, min %.2fms\n", deltaTime * 1000.0, minDeltaTime * 1000.0);
    }
    frameIndex++;
#endif
#if DEBUG_STATS
    printf("Total Triangle Idx Count %u\n", triIndexCount);
    printf("Max Triangle Idx Count Per Task %u\n", maxTaskTriCount);
#endif
#if VERIFY_TRAVERSAL
    printf("Traversal imp/ref blockStackSum, tileTriSum: %d/%d, %d/%d\n",
        int(impBlockStackSum), int(refBlockStackSum),
        int(impTileTriSum), int(refTileTriSum));
#endif
    return triIndexCount;
}

} // namespace hvvr
