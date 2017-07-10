/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model_import.h"
#include "raycaster.h"

#include <string.h>

namespace model_import {

bool compareExtension(const char* a, const char* b) {
    if (_stricmp(a, b) == 0)
        return true;
    return false;
}

bool load(const char* path, Model& model) {
    const char* extension = strrchr(path, '.');
    if (extension == nullptr)
        return false;
    extension++;

    if (compareExtension(extension, "bin"))
    {
        return loadBin(path, model);
    }
#if MODEL_IMPORT_ENABLE_FBX
    else if (compareExtension(extension, "fbx"))
    {
        return loadFbx(path, model);
    }
#endif

    return false;
}

bool createObjects(hvvr::Raycaster& raycaster, Model& model) {
    for (const Texture& texture : model.textures) {
        hvvr::Texture* object = raycaster.createTexture(texture.tex);
        if (object == nullptr)
            return false;
    }

    for (const hvvr::LightUnion& light : model.lights) {
        hvvr::Light* object = raycaster.createLight(light);
        if (object == nullptr)
            return false;
    }

    for (const Mesh& mesh : model.meshes) {
        hvvr::Model* object = raycaster.createModel(mesh.data);
        if (object == nullptr)
            return false;

        object->setTransform(mesh.transform);
    }

    return true;
}

} // namespace model_import
