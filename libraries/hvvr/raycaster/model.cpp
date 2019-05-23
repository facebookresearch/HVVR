/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "graphics_types.h"
#include "material.h"
#include "raycaster.h"
#include "vector_math.h"

#include <assert.h>


namespace hvvr {

Model::Model(Raycaster& raycaster, MeshData&& meshData)
    : _raycaster(raycaster), _meshData(std::move(meshData)), _transform(transform::identity()) {}

Model::~Model() {}

void Model::setTransform(const transform& transform) {
    _transform = transform;

    _raycaster.setSceneDirty();
}

} // namespace hvvr
