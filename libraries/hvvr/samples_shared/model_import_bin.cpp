/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model_import.h"

#include <stdio.h>

namespace model_import {

struct BinHeader {
    uint32_t version = 1;
    uint32_t textureCount;
    uint32_t lightCount;
    uint32_t meshCount;
};

struct BinTexture {
    hvvr::TextureFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t stride;

    BinTexture() {}
    explicit BinTexture(const Texture& tex)
        : format(tex.tex.format), width(tex.tex.width), height(tex.tex.height), stride(tex.tex.strideElements) {}

    explicit operator Texture() const {
        hvvr::TextureData texDesc = {};
        texDesc.format = format;
        texDesc.width = width;
        texDesc.height = height;
        texDesc.strideElements = stride;

        return Texture(texDesc);
    }
};

struct BinMesh {
    hvvr::transform transform;
    uint32_t vertexCount;
    uint32_t triangleCount;
    uint32_t nodeCount;
    uint32_t materialCount;

    BinMesh() {}
    explicit BinMesh(const Mesh& mesh)
        : transform(mesh.transform)
        , vertexCount(uint32_t(mesh.data.verts.size()))
        , triangleCount(uint32_t(mesh.data.triShade.size()))
        , nodeCount(uint32_t(mesh.data.nodes.size()))
        , materialCount(uint32_t(mesh.data.materials.size())) {}

    explicit operator Mesh() const {
        Mesh rval;
        rval.transform = transform;
        rval.data.verts.resize(vertexCount);
        rval.data.triShade.resize(triangleCount);
        rval.data.nodes.resize(nodeCount);
        rval.data.materials.resize(materialCount);
        return rval;
    }
};

bool loadBin(const char* path, Model& model) {
    bool ok = false;
    FILE* file = fopen(path, "rb");
    {
        if (!file) {
            goto load_bin_fail;
        }

        BinHeader header;
        fread(&header, sizeof(header), 1, file);
        if (header.version != BinHeader().version) {
            printf("error: bin file version mismatch\n");
            goto load_bin_fail;
        }

        model.textures.reserve(header.textureCount);
        for (uint32_t n = 0; n < header.textureCount; n++) {
            BinTexture binTex;
            fread(&binTex, sizeof(binTex), 1, file);
            Texture tex(binTex);

            size_t sizeBytes = hvvr::getTextureSize(tex.tex.strideElements, tex.tex.height, tex.tex.format);
            tex.tex.data = new uint8_t [sizeBytes];
            fread((void*)tex.tex.data, sizeBytes, 1, file);

            model.textures.emplace_back(std::move(tex));
        }

        model.lights.reserve(header.lightCount);
        for (uint32_t n = 0; n < header.lightCount; n++) {
            hvvr::LightUnion light;
            fread(&light, sizeof(light), 1, file);

            model.lights.emplace_back(std::move(light));
        }

        model.meshes.reserve(header.meshCount);
        for (uint32_t n = 0; n < header.meshCount; n++) {
            BinMesh binMesh;
            fread(&binMesh, sizeof(binMesh), 1, file);
            Mesh mesh(binMesh);

            fread(mesh.data.verts.data(), sizeof(hvvr::ShadingVertex) * binMesh.vertexCount, 1, file);
            fread(mesh.data.triShade.data(), sizeof(hvvr::PrecomputedTriangleShade) * binMesh.triangleCount, 1, file);
            fread(mesh.data.nodes.data(), sizeof(hvvr::TopologyNode) * binMesh.nodeCount, 1, file);
            fread(mesh.data.materials.data(), sizeof(hvvr::SimpleMaterial) * binMesh.materialCount, 1, file);

            model.meshes.emplace_back(std::move(mesh));
        }
    }

    ok = true;
load_bin_fail:

    if (file) {
        fclose(file);
    }
    return ok;
}

bool saveBin(const char* path, const Model& model) {
    FILE* file = fopen(path, "wb");
    if (!file)
        return false;

    BinHeader header;
    header.textureCount = uint32_t(model.textures.size());
    header.lightCount = uint32_t(model.lights.size());
    header.meshCount = uint32_t(model.meshes.size());
    fwrite(&header, sizeof(header), 1, file);

    for (uint32_t n = 0; n < header.textureCount; n++) {
        const Texture& tex = model.textures[n];
        BinTexture binTex(tex);
        size_t sizeBytes = hvvr::getTextureSize(binTex.stride, binTex.height, binTex.format);

        fwrite(&binTex, sizeof(binTex), 1, file);
        fwrite(tex.tex.data, sizeBytes, 1, file);
    }

    for (uint32_t n = 0; n < header.lightCount; n++) {
        const hvvr::LightUnion& light = model.lights[n];

        fwrite(&light, sizeof(light), 1, file);
    }

    for (uint32_t n = 0; n < header.meshCount; n++) {
        const Mesh& mesh = model.meshes[n];
        BinMesh binMesh(mesh);

        fwrite(&binMesh, sizeof(binMesh), 1, file);
        fwrite(mesh.data.verts.data(), sizeof(hvvr::ShadingVertex) * binMesh.vertexCount, 1, file);
        fwrite(mesh.data.triShade.data(), sizeof(hvvr::PrecomputedTriangleShade) * binMesh.triangleCount, 1, file);
        fwrite(mesh.data.nodes.data(), sizeof(hvvr::TopologyNode) * binMesh.nodeCount, 1, file);
        fwrite(mesh.data.materials.data(), sizeof(hvvr::SimpleMaterial) * binMesh.materialCount, 1, file);
    }

    fclose(file);
    return true;
}

} // namespace model_import
