#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "material.h"
#include "shading_helpers.h"
#include "vector_math.h"


namespace hvvr {

CUDA_DEVICE_INL vector4 ReadTexture(cudaTextureObject_t* tex, unsigned index, vector2 uv, vector2 ddx, vector2 ddy) {
    return vector4(tex2DGrad<float4>(tex[index], uv.x, uv.y, float2(ddx), float2(ddy)));
}

template <typename PrecomputedTriangleType>
CUDA_DEVICE_INL vector4 BarycentricShade(float b[3],
                                         uint32_t,
                                         const PrecomputedTriangleType&,
                                         vector3,
                                         vector3,
                                         const ShadingVertex* CUDA_RESTRICT,
                                         const SimpleMaterial* CUDA_RESTRICT,
                                         cudaTextureObject_t*) {
    return vector4(b[0], b[1], b[2], 1.0f);
}

CUDA_DEVICE_INL vector4 hashedColor(uint32_t hash) {
    float red = sinf(float(hash) * 234.034f) * 0.5f + 0.5f;
    float green = sinf(float(hash) * 64.4398f) * 0.5f + 0.5f;
    float blue = sinf(float(hash) * 0.02134f) * 0.5f + 0.5f;
    return vector4(red, green, blue, 1.0f);
}

template <typename PrecomputedTriangleType>
CUDA_DEVICE_INL vector4 UVShade(float b[3],
                                uint32_t,
                                const PrecomputedTriangleType& tri,
                                vector3,
                                vector3,
                                const ShadingVertex* CUDA_RESTRICT verts,
                                const SimpleMaterial* CUDA_RESTRICT,
                                cudaTextureObject_t*) {
    vector3 result = vector3(0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        ShadingVertex vert = verts[tri.indices[i]];
        result += vector3(vert.uv * b[i], 0.0);
    }
    return vector4(result, 1.0f);
}

template <typename PrecomputedTriangleType>
CUDA_DEVICE_INL vector4 WSNormalShade(float b[3],
                                      uint32_t,
                                      const PrecomputedTriangleType& tri,
                                      vector3,
                                      vector3,
                                      const ShadingVertex* CUDA_RESTRICT verts,
                                      const SimpleMaterial* CUDA_RESTRICT,
                                      cudaTextureObject_t*) {
    vector3 normal = vector3(0.0, 0.0, 0.0);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        ShadingVertex vert = verts[tri.indices[i]];
        normal += vector3(vert.normal) * b[i];
    }
    return vector4(normal * 0.5f + 0.5f, 1.0f);
}

template <typename PrecomputedTriangleType>
CUDA_DEVICE_INL vector4 NoMaterialBRDFShade(float b[3],
                                            uint32_t,
                                            const PrecomputedTriangleType& tri,
                                            vector3 cameraPos,
                                            vector3 cameraLookVector,
                                            const ShadingVertex* CUDA_RESTRICT verts,
                                            const SimpleMaterial* CUDA_RESTRICT,
                                            cudaTextureObject_t*) {
    vector3 position = vector3(0.0, 0.0, 0.0);
    vector3 normal = vector3(0.0, 0.0, 0.0);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        ShadingVertex vert = verts[tri.indices[i]];
        position += vert.pos * b[i];
        normal += vector3(vert.normal) * b[i];
    }
    LightPoint light = DefaultPointLight(vector3(0.0f, 2.0f, 0.0f));

    vector3 V = normalize(cameraPos - position);
    vector3 BaseColor = vector3(1.0f, 1.0f, 1.0f);
    // F0 of nonmetals is constant 0.04f in Unreal
    vector3 F0 = vector3(0.04f);
    float roughness = 1.0f;
    return vector4(PointLightContribution(position, normal, V, BaseColor, roughness, F0, light), 1.0f);
}

template <typename PrecomputedTriangleType>
CUDA_DEVICE_INL vector4 LambertianTextureShade(float b[3],
                                               uint32_t,
                                               const PrecomputedTriangleType& tri,
                                               vector3 cameraPos,
                                               vector3 cameraLookVector,
                                               const ShadingVertex* CUDA_RESTRICT verts,
                                               const SimpleMaterial* CUDA_RESTRICT materials,
                                               cudaTextureObject_t* textures) {
    vector2 uv = vector2(0.0f);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        ShadingVertex vert = verts[tri.indices[i]];
        uv += vert.uv * b[i];
    }
    SimpleMaterial material = materials[tri.material];
    uint64_t temp = material.textureIDsAndShadingModel;
    uint32_t textureIndex = (temp >> SimpleMaterial::diffuseBitShift) & SimpleMaterial::textureMask;
    return ReadTexture(textures, textureIndex, uv, vector2(0.0f), vector2(0.0f));
}

template <typename PrecomputedTriangleType>
CUDA_DEVICE_INL vector4 WSNormalAndCSZShade(float b[3],
                                            uint32_t,
                                            const PrecomputedTriangleType& tri,
                                            vector3 cameraPos,
                                            vector3 cameraLookVector,
                                            const ShadingVertex* CUDA_RESTRICT verts,
                                            const SimpleMaterial* CUDA_RESTRICT materials,
                                            cudaTextureObject_t* textures) {
    vector3 position = vector3(0.0, 0.0, 0.0);
    vector3 normal = vector3(0.0, 0.0, 0.0);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        ShadingVertex vert = verts[tri.indices[i]];
        position += vert.pos * b[i];
        normal += vector3(vert.normal) * b[i];
    }

    float csZ = -dot(cameraLookVector, position - cameraPos);
    return vector4(normal, csZ);
}

CUDA_DEVICE_INL vector4 FullGGXShadeAfterInterpolation(uint32_t materialIndex,
                                                       vector2 uv,
                                                       vector2 ddx,
                                                       vector2 ddy,
                                                       vector3 position,
                                                       vector3 normal,
                                                       vector3 cameraPos,
                                                       vector3 cameraLookVector,
                                                       const ShadingVertex* CUDA_RESTRICT verts,
                                                       const SimpleMaterial* CUDA_RESTRICT materials,
                                                       cudaTextureObject_t* textures,
                                                       const LightingEnvironment& env) {
    SimpleMaterial material = materials[materialIndex];
    uint64_t temp = material.textureIDsAndShadingModel;

    ShadingModel shadingModel = ShadingModel(temp & SimpleMaterial::textureMask);
    if (shadingModel == ShadingModel::emissive) {
        uint32_t emissiveTextureIndex = (temp >> SimpleMaterial::diffuseBitShift) & SimpleMaterial::textureMask;

        vector4 emissiveColor = material.emissive;
        if (emissiveTextureIndex != SimpleMaterial::badTextureIndex) {
            emissiveColor = ReadTexture(textures, emissiveTextureIndex, uv, ddx, ddy);
        }

        return emissiveColor;
    } else { // Phong which is GGX...
        uint32_t diffuseTextureIndex = (temp >> SimpleMaterial::diffuseBitShift) & SimpleMaterial::textureMask;
        uint32_t specularTextureIndex = (temp >> SimpleMaterial::specularBitShift) & SimpleMaterial::textureMask;
        uint32_t glossyTextureIndex = (temp >> SimpleMaterial::glossyBitShift) & SimpleMaterial::textureMask;

        vector3 BaseColor = vector3(material.diffuse);
        if (diffuseTextureIndex != SimpleMaterial::badTextureIndex) {
            BaseColor = vector3(ReadTexture(textures, diffuseTextureIndex, uv, ddx, ddy));
        }

        vector3 V = normalize(cameraPos - position);

        vector3 F0 = vector3(material.specular);
        if (specularTextureIndex != SimpleMaterial::badTextureIndex) {
            F0 = vector3(ReadTexture(textures, specularTextureIndex, uv, ddx, ddy));
        }

        float roughness = 1.0 - material.glossiness;
        if (glossyTextureIndex != SimpleMaterial::badTextureIndex) {
            roughness = 1.0f - ReadTexture(textures, glossyTextureIndex, uv, ddx, ddy).x;
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

            radiance +=
                DirectionalLightContribution(position, normal, V, BaseColor, roughness, F0, env.directionalLights[i]);
        }

        for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
            if (i >= env.pointLightCount)
                break;

            radiance += PointLightContribution(position, normal, V, BaseColor, roughness, F0, env.pointLights[i]);
        }

        for (int i = 0; i < MAX_SPOT_LIGHTS; ++i) {
            if (i >= env.spotLightCount)
                break;

            radiance += SpotLightContribution(position, normal, V, BaseColor, roughness, F0, env.spotLights[i]);
        }

        return vector4(max(radiance.x, 0.0f), max(radiance.y, 0.0f), max(radiance.z, 0.0f), 1.0f);
    }
}

template <typename PrecomputedTriangleType>
CUDA_DEVICE_INL vector4 FullGGXShade(float b[3],
                                     float bOffX[3],
                                     float bOffY[3],
                                     uint32_t,
                                     const PrecomputedTriangleType& tri,
                                     vector3 cameraPos,
                                     vector3 cameraLookVector,
                                     const ShadingVertex* CUDA_RESTRICT verts,
                                     const SimpleMaterial* CUDA_RESTRICT materials,
                                     cudaTextureObject_t* textures,
                                     const LightingEnvironment& env) {
    vector2 uv(0.0f);
    vector2 ddx(0.0f);
    vector2 ddy(0.0f);
    vector3 position(0.0f);
    vector3 normal(0.0f);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        ShadingVertex vert = verts[tri.indices[i]];
        position += vert.pos * b[i];
        normal += vector3(vert.normal) * b[i];
        uv += vert.uv * b[i];
        ddx += vert.uv * bOffX[i];
        ddy += vert.uv * bOffY[i];
    }
    ddx -= uv;
    ddy -= uv;
    return FullGGXShadeAfterInterpolation(tri.material, uv, ddx, ddy, position, normalize(normal), cameraPos,
                                          cameraLookVector, verts, materials, textures, env);
}

} // namespace hvvr
