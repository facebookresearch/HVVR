/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "light.h"
#include "model.h"
#include "raycaster.h"
#include "texture.h"

#include <assert.h>


namespace hvvr {

void Raycaster::cleanupScene() {
    _textures.clear();
    _lights.clear();
    _models.clear();

    DestroyAllTextures();
}


Texture* Raycaster::createTexture(const TextureData& textureData) {
    _textures.emplace_back(std::make_unique<Texture>(textureData));
    return (_textures.end() - 1)->get();
}

void Raycaster::destroyTexture(Texture* texture) {
    for (auto it = _textures.begin(); it != _textures.end(); ++it) {
        if (it->get() == texture) {
            _textures.erase(it);
            return;
        }
    }
    assert(false); // not found
}

Light* Raycaster::createLight(const LightUnion& lightUnion) {
    _lights.emplace_back(std::make_unique<Light>(lightUnion));
    return (_lights.end() - 1)->get();
}

void Raycaster::destroyLight(Light* light) {
    for (auto it = _lights.begin(); it != _lights.end(); ++it) {
        if (it->get() == light) {
            _lights.erase(it);
            return;
        }
    }
    assert(false); // not found
}

Model* Raycaster::createModel(MeshData&& meshData) {
    _sceneDirty = true;

    _models.emplace_back(std::make_unique<Model>(*this, std::move(meshData)));
    return (_models.end() - 1)->get();
}
Model* Raycaster::createModel(const MeshData& meshData) {
    return createModel(std::move(MeshData(meshData)));
}

void Raycaster::destroyModel(Model* model) {
    for (auto it = _models.begin(); it != _models.end(); ++it) {
        if (it->get() == model) {
            _models.erase(it);
            return;
        }
    }
    assert(false); // not found
}

void Raycaster::setSceneDirty() {
    _sceneDirty = true;
}

} // namespace hvvr
