#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <host_defines.h>

#if defined(__CUDA_ARCH__)
// we're compiling for the GPU target
# define CUDA_COMPILE 1
# define CUDA_ARCH __CUDA_ARCH__
#else
# define CUDA_COMPILE 0
# define CUDA_ARCH 0
#endif

#define CUDA_DEVICE __device__
#define CUDA_DEVICE_INL inline CUDA_DEVICE
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_HOST_DEVICE_INL inline CUDA_HOST_DEVICE
#define CUDA_KERNEL __global__

#define CUDA_RESTRICT __restrict__

// shortened macros for code with a lot of definitions (vector_math.h, for example)
#define CDI CUDA_DEVICE_INL
#define CHD CUDA_HOST_DEVICE
#define CHDI CUDA_HOST_DEVICE_INL

#define WARP_SIZE 32
