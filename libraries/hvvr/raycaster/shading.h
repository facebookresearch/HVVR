#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "material.h"
#include "shading_helpers.h"
#include "vector_math.h"


namespace hvvr {

template <typename T>
CUDA_DEVICE_INL T interpolate(const T& v0, const T& v1, const T& v2, vector3 b) {
    return v0 * b.x + v1 * b.y + v2 * b.z;
}

CUDA_DEVICE_INL InterpolatedVertex interpolate(const ShadingVertex* CUDA_RESTRICT verts,
                                               const PrecomputedTriangleShade& triShade,
                                               vector3 b) {
    ShadingVertex v0 = verts[triShade.indices[0]];
    ShadingVertex v1 = verts[triShade.indices[1]];
    ShadingVertex v2 = verts[triShade.indices[2]];

    InterpolatedVertex r;
    r.pos = interpolate(v0.pos, v1.pos, v2.pos, b);
    r.normal = interpolate(vector3(v0.normal), vector3(v1.normal), vector3(v2.normal), b);
    r.uv = interpolate(v0.uv, v1.uv, v2.uv, b);

    return r;
}

CUDA_DEVICE_INL vector4 hashedColor(uint32_t hash) {
    float red = sinf(float(hash) * 234.034f) * 0.5f + 0.5f;
    float green = sinf(float(hash) * 64.4398f) * 0.5f + 0.5f;
    float blue = sinf(float(hash) * 0.02134f) * 0.5f + 0.5f;
    return vector4(red, green, blue, 1.0f);
}

CUDA_DEVICE_INL vector4 ReadTexture(cudaTextureObject_t* tex, unsigned index, vector2 uv, vector2 ddx, vector2 ddy) {
    return vector4(tex2DGrad<float4>(tex[index], uv.x, uv.y, float2(ddx), float2(ddy)));
}

CUDA_DEVICE_INL vector4 BarycentricShade(vector3 b) {
    return vector4(b, 1.0f);
}

CUDA_DEVICE_INL vector4 UVShade(const InterpolatedVertex& vInterp) {
    return vector4(vInterp.uv, 0.0f, 1.0f);
}

CUDA_DEVICE_INL vector4 WSNormalShade(const InterpolatedVertex& vInterp) {
    return vector4(vInterp.normal * 0.5f + 0.5f, 1.0f);
}

CUDA_DEVICE_INL vector4 NoMaterialBRDFShade(const InterpolatedVertex& vInterp,
                                            vector3 cameraPos) {
    LightPoint light = DefaultPointLight(vector3(0.0f, 2.0f, 0.0f));
    vector3 normal = normalize(vInterp.normal);

    vector3 V = normalize(cameraPos - vInterp.pos);
    vector3 BaseColor = vector3(1.0f, 1.0f, 1.0f);
    // F0 of nonmetals is constant 0.04f in Unreal
    vector3 F0 = vector3(0.04f);
    float roughness = 1.0f;
    return vector4(PointLightContribution(vInterp.pos, normal, V, BaseColor, roughness, F0, light), 1.0f);
}

CUDA_DEVICE_INL vector4 LambertianTextureShade(uint32_t materialIndex,
                                               const InterpolatedVertex& vInterp,
                                               const SimpleMaterial* CUDA_RESTRICT materials,
                                               cudaTextureObject_t* textures) {
    SimpleMaterial material = materials[materialIndex];
    uint64_t temp = material.textureIDsAndShadingModel;
    uint32_t textureIndex = (temp >> SimpleMaterial::diffuseBitShift) & SimpleMaterial::textureMask;
    return ReadTexture(textures, textureIndex, vInterp.uv, vector2(0.0f), vector2(0.0f));
}

CUDA_DEVICE_INL vector4 GGXShade(uint32_t materialIndex,
                                 const InterpolatedVertex& vInterp,
                                 vector2 dUVdX,
                                 vector2 dUVdY,
                                 vector3 cameraPos,
                                 const SimpleMaterial* CUDA_RESTRICT materials,
                                 cudaTextureObject_t* textures,
                                 const LightingEnvironment& env) {
    SimpleMaterial material = materials[materialIndex];
    uint64_t temp = material.textureIDsAndShadingModel;

    vector3 normal = normalize(vInterp.normal);

    ShadingModel shadingModel = ShadingModel(temp & SimpleMaterial::textureMask);
    if (shadingModel == ShadingModel::emissive) {
        uint32_t emissiveTextureIndex = (temp >> SimpleMaterial::diffuseBitShift) & SimpleMaterial::textureMask;

        vector4 emissiveColor = material.emissive;
        if (emissiveTextureIndex != SimpleMaterial::badTextureIndex) {
            emissiveColor = ReadTexture(textures, emissiveTextureIndex, vInterp.uv, dUVdX, dUVdY);
        }

        return emissiveColor;
    } else { // Phong which is GGX...
        uint32_t diffuseTextureIndex = (temp >> SimpleMaterial::diffuseBitShift) & SimpleMaterial::textureMask;
        uint32_t specularTextureIndex = (temp >> SimpleMaterial::specularBitShift) & SimpleMaterial::textureMask;
        uint32_t glossyTextureIndex = (temp >> SimpleMaterial::glossyBitShift) & SimpleMaterial::textureMask;

        vector3 BaseColor = vector3(material.diffuse);
        if (diffuseTextureIndex != SimpleMaterial::badTextureIndex) {
            BaseColor = vector3(ReadTexture(textures, diffuseTextureIndex, vInterp.uv, dUVdX, dUVdY));
        }

        vector3 V = normalize(cameraPos - vInterp.pos);

        vector3 F0 = vector3(material.specular);
        if (specularTextureIndex != SimpleMaterial::badTextureIndex) {
            F0 = vector3(ReadTexture(textures, specularTextureIndex, vInterp.uv, dUVdX, dUVdY));
        }

        float roughness = 1.0 - material.glossiness;
        if (glossyTextureIndex != SimpleMaterial::badTextureIndex) {
            roughness = 1.0f - ReadTexture(textures, glossyTextureIndex, vInterp.uv, dUVdX, dUVdY).x;
        }

        vector3 radiance = vector3(material.emissive);

        // ambient
        radiance += BaseColor * vector3(.15f, .15f, .165f);

        // Give the compiler a hint so it can unroll the loop and avoid attempting to
        // dynamically index the LightingEnvironment constants, which would force a local
        // memory spill.
        for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; ++i) {
            if (i >= env.directionalLightCount)
                break;

            radiance += DirectionalLightContribution(vInterp.pos, normal, V, BaseColor, roughness, F0,
                                                     env.directionalLights[i]);
        }

        for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
            if (i >= env.pointLightCount)
                break;

            radiance += PointLightContribution(vInterp.pos, normal, V, BaseColor, roughness, F0, env.pointLights[i]);
        }

        for (int i = 0; i < MAX_SPOT_LIGHTS; ++i) {
            if (i >= env.spotLightCount)
                break;

            radiance += SpotLightContribution(vInterp.pos, normal, V, BaseColor, roughness, F0, env.spotLights[i]);
        }

        return vector4(max(radiance.x, 0.0f), max(radiance.y, 0.0f), max(radiance.z, 0.0f), 1.0f);
    }
}

} // namespace hvvr
