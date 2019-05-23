#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_decl.h"

// http://stereopsis.com/radix.html
CUDA_DEVICE_INL uint32_t FloatFlip(uint32_t f) {
    uint32_t mask = -int32_t(f >> 31) | 0x80000000;
    return f ^ mask;
}
CUDA_DEVICE_INL uint32_t IFloatFlip(uint32_t f) {
    // TODO: CUDA 8.0 -> 9.1 transition... PTX is OK (unchanged), SASS is busted
    //uint32_t mask = ((f >> 31) - 1) | 0x80000000;

    // this seems to work on 9.1
    uint32_t mask = ((f & 0x80000000) == 0) ? 0xffffffff : 0x80000000;

    return f ^ mask;
}
CUDA_DEVICE_INL uint32_t FloatFlipF(float f) {
    return FloatFlip(__float_as_int(f));
}
CUDA_DEVICE_INL float IFloatFlipF(uint32_t f) {
    return __int_as_float(IFloatFlip(f));
}

// bitonic sort within a single thread, see:
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/Data-Parallel_Algorithms.html ("Bitonic Sort" sample, bitonic_kernel.cu)
// https://github.com/Microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/ParticleTileCullingCS.hlsl
// http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
// If count is known at compile time, this should be optimized down to a series
// of min/max ops without branching.
template <typename T>
CUDA_DEVICE_INL void sortBitonic(T* sortKeys, int count) {
	// find the power of two upper bound on count to determine how many iterations we need
    int countPow2 = count;
    if (__popc(count) > 1) {
        countPow2 = (1 << (32 - __clz(count)));
    }

    for (int k = 2; k <= countPow2; k *= 2) {
        // align up to the current power of two
        count = (count + k - 1) & ~(k - 1);

		// merge
        for (int j = k / 2; j > 0; j /= 2) {
			// for each pair of elements
            for (int i = 0; i < count / 2; i++) {
				// find the pair of elements to compare
                int mask = j - 1;
                int s0 = ((i & ~mask) << 1) | (i & mask);
                int s1 = s0 | j;

                T a = sortKeys[s0];
                T b = sortKeys[s1];

                bool compare = (a < b);
                bool direction = ((s0 & k) == 0);

                if (compare != direction) {
                    sortKeys[s0] = b;
                    sortKeys[s1] = a;
                }
            }
        }
    }
}
