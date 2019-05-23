#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_decl.h"

CUDA_DEVICE_INL int laneGetIndex() {
    return threadIdx.x % WARP_SIZE;
}

CUDA_DEVICE_INL int laneGetMaskAll() {
    return ~uint32_t(0);
}
CUDA_DEVICE_INL int laneGetMaskLT() {
    return (1 << laneGetIndex()) - 1;
}

template <typename T>
CUDA_DEVICE_INL T laneBroadcast(T v, int laneIndex) {
    return __shfl_sync(__activemask(), v, laneIndex);
}

// avoid calling this multiple times for the same value of pred
// the compiler doesn't like to optimize this intrinsic
CUDA_DEVICE_INL int warpBallot(bool pred) {
    return __ballot_sync(__activemask(), pred);
}

CUDA_DEVICE_INL int warpGetFirstActiveIndex(int predMask) {
    return __ffs(predMask) - 1;
}

// Instead of using one atomic per active thread to append into a compacted buffer,
// this uses one atomic per warp. Also, relative ordering within a warp is preserved.
// Input is a predicate indicating whether a thread needs an output slot or not,
// and a pointer to a (shared or memory) location with the running count to increment.
// Output is a per-thread output index, as if each active thread where pred is true had
// performed an atomic increment on counter.
template <typename T>
CUDA_DEVICE_INL T warpAppend(bool pred, T* counter) {
    int predMask = warpBallot(pred);
    int firstActive = warpGetFirstActiveIndex(predMask);
    bool isFirstActive = (laneGetIndex() == firstActive);

    T outputIndexBase;
    if (isFirstActive) {
        T outputCount = __popc(predMask);
        outputIndexBase = atomicAdd(counter, outputCount);
    }
    // broadcast result from firstActive to all lanes
    outputIndexBase = laneBroadcast(outputIndexBase, firstActive);

    T outputIndex = outputIndexBase + __popc(predMask & laneGetMaskLT());

    return outputIndex;
}

// butterfly reduction
// All lanes will contain the same result, as if it had been broadcast.
// Inactive lanes will contribute undefined results, breaking the reduction...
// so ensure all lanes are active (uniform branching only).
template <typename T, typename Op>
CUDA_DEVICE_INL T warpReduce(T val, const Op& op) {
    for (int xorMask = WARP_SIZE / 2; xorMask >= 1; xorMask /= 2)
        val = op(val, __shfl_xor_sync(__activemask(), val, xorMask));
    return val;
}
