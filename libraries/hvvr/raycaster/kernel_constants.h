#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "raycaster_common.h"

#define MSAA_SHADE 0
#define SSAA_SHADE 1
#define SUPERSHADING_MODE MSAA_SHADE
#define JITTER_SAMPLES 0
// TODO(anankervis): GPU foveated path doesn't pass along correct tile culling frusta to intersect (yet)
#define FOVEATED_TRIANGLE_FRUSTA_TEST_DISABLE 0

#define DOF_LENS_POS_LOOKUP_TABLE_TILES 4

#define ENABLE_HACKY_WIDE_FOV 0
#define HACKY_WIDE_FOV_W 210.0f
#define HACKY_WIDE_FOV_H 130.0f

#define SM_BARYCENTRIC 2
#define SM_TRI_ID 3
#define SM_UV 4
#define SM_WS_NORMAL 5
#define SM_NO_MATERIAL_BRDF 6
#define SM_LAMBERTIAN_TEXTURE 7
#define SM_FULL_BRDF 8
#define SM_MATERIAL_ID 9
#define COLOR_SHADING_MODE SM_FULL_BRDF

#define CUDA_BLOCK_WIDTH 16
#define CUDA_BLOCK_HEIGHT 8
#define CUDA_BLOCK_SIZE (CUDA_BLOCK_WIDTH * CUDA_BLOCK_HEIGHT)
