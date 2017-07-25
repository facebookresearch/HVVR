#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// TODO: Remove all dependencies on DX
#ifdef _WIN32
#define DX_SUPPORTED
#endif

// TODO(anankervis): move this elsewhere - the raycaster core shouldn't care what the output mode is
#define OUTPUT_MODE_NONE 0
#define OUTPUT_MODE_3D_API 1
// not yet implemented
#define OUTPUT_MODE_PNG 2
#define OUTPUT_MODE OUTPUT_MODE_3D_API

namespace hvvr {

// Rays per tile
// If tile culling is taking a long time or you have too many
// triangle indices to ship to the GPU every frame, increase this.
// Prefer powers of 2.
static const unsigned TILE_SIZE = 128;

static const unsigned TILES_PER_BLOCK = 64;
static const unsigned BLOCK_SIZE = TILE_SIZE * TILES_PER_BLOCK;

static const unsigned MAX_DIRECTIONAL_LIGHTS = 1;
static const unsigned MAX_POINT_LIGHTS = 4;
static const unsigned MAX_SPOT_LIGHTS = 4;

// max supported is 32x
static const unsigned COLOR_MODE_MSAA_RATE = 16;

} // namespace hvvr
