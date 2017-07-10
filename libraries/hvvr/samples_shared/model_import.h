#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector_math.h"
#include "light.h"
#include "model.h"
#include "texture.h"

#include <vector>

#define MODEL_IMPORT_ENABLE_FBX 0

namespace hvvr {
class Raycaster;
}

namespace model_import {

struct Texture {
    hvvr::TextureData tex;

    Texture() {
        tex.data = nullptr;
    }
    Texture(Texture&& t) {
        tex = std::move(t.tex);
        t.tex.data = nullptr;
    }

    explicit Texture(hvvr::TextureData& _tex) : tex(_tex) {}

    ~Texture() {
        if (tex.data)
            delete [] tex.data;
        tex.data = nullptr;
    }
};

struct Mesh {
    hvvr::transform transform;
    hvvr::MeshData data;
};

struct Model {
    std::vector<Texture> textures;
    std::vector<hvvr::LightUnion> lights;
    std::vector<Mesh> meshes;
};

// determine file type based on file extension
bool load(const char* path, Model& model);

bool loadBin(const char* path, Model& model);
bool saveBin(const char* path, const Model& model);

#if MODEL_IMPORT_ENABLE_FBX
bool loadFbx(const char* path, Model& model);
#endif

bool createObjects(hvvr::Raycaster& raycaster, Model& model);

} // namespace model_import
