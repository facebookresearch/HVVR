#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "graphics_types.h"
#include "material.h"
#include "vector_math.h"

#include <vector>


namespace gfx {
struct TopologyNode;
} // namespace gfx

namespace hvvr {

class Raycaster;

struct MeshData {
    std::vector<ShadingVertex> verts;
    std::vector<PrecomputedTriangleShade> triShade;
    std::vector<TopologyNode> nodes;
    std::vector<SimpleMaterial> materials;
};

class Model {
public:
    Model(Raycaster& raycaster, MeshData&& meshData);
    ~Model();

    Model(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;

    const MeshData& getMesh() const {
        return _meshData;
    }

    void setTransform(const transform& transform);
    const transform& getTransform() const {
        return _transform;
    }

protected:
    Raycaster& _raycaster;

    MeshData _meshData;
    transform _transform;
};

} // namespace hvvr
