#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "constants_math.h"
#include "cuda_decl.h"
#include "light.h"
#include "lighting_env.h"
#include "util_graphics.h"
#include "vector_math.h"

namespace hvvr {

CDI vector3 fastSaturate(vector3 color) {
    return vector3(__saturatef(color.x), __saturatef(color.y), __saturatef(color.z));
}

CDI LightPoint DefaultPointLight(vector3 pos) {
    return {pos, 10.0f, vector3(3.0f), 1.0f};
}

CDI float square(float x) {
    return x * x;
}

CDI float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

// Schlick approximation to Fresnel reflectance for microfacet BRDFs (Taken from Renderer/Lighting.hlsli)
CDI vector3 FresnelSchlick(vector3 SpecColor, vector3 L, vector3 H) {
    float LdotH = __saturatef(dot(L, H));
    return SpecColor + (vector3(1.0f) - SpecColor) * pow5(1.0f - LdotH);
}

// Normalized Blinn-Phong BRDF with the Schlick approximation to Fresnel reflectance (Taken from
// Renderer/Lighting.hlsli)
CDI vector3 BlinnPhongBRDF(vector3 N, vector3 L, vector3 V, vector3 DiffuseColor, vector3 SpecColor, float SpecPower) {
    // Diffuse
    float NdotL = __saturatef(dot(N, L));

    DiffuseColor = DiffuseColor * NdotL;

    // Specular
    vector3 H = normalize(L + V);
    float NdotH = __saturatef(dot(N, H));

    SpecColor = FresnelSchlick(SpecColor, L, H) * (SpecPower + 2.0) / 8.0 * pow(NdotH, SpecPower) * NdotL;

    return DiffuseColor + SpecColor;
}

/** This function does not multiply by NdotV, so must multiply later */
CDI float VisibilitySchlickSmith_G1V(float NdotV, float k) {
    return 1.0f / (NdotV * (1.0f - k) + k);
}

// General Microfacet Scattering Function:
// For a point X with macrosurface normal n, incident direction w_i, and outgoing direction w_o, the general BRDF is:
// f_X,n(w_i,w_o) = A_X(w_i,w_o) / pi + (D_X,n(w_h)F_X(w_i,w_o)G_X(w_i,w_o)/(4*(dot(n,w_i)*dot(n,w_o))
//
// Where w_h is the half vector between w_i and w_o
//
// A_X is the shallow-subsurface scattering function, commonly approximated as Lambertian A_X(w_i, w_o) = l_x
// D is the microfacet orientation distribution
// F is fresnel
// G is the geometric shadowing and masking term
//
// John Hable claimed the industry standard (in 2014) was
// D = GGX Distribution (sidenote: GGX doesn't stand for anything sensible)
// F = Schlick's Fresnel approximation
// V = Schlick approximation of Smith solved for GGX
// http://www.filmicworlds.com/2014/04/21/optimizing-ggx-shaders-with-dotlh/
// Unreal uses GGX for D,
// modified Schlick's approximation for GGX (to closer match Smith) for G,
// and a Spherical Guassian approximation of Schlick's approximation for F
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
// Unity uses GGX for D as of 1/2016: https://blogs.unity3d.com/2016/01/25/ggx-in-unity-5-3/
//
//
// We'll use John Hable's choices for now. They are very close to Unreal's.
CDI vector3 GGX_SchlickFresnel_SchlickSmithGGX(vector3 N, vector3 L, vector3 V, float roughness, vector3 F0) {
    float alpha = roughness * roughness;
    vector3 H = normalize(L + V);

    // D = GGX
    float alpha2 = square(alpha);
    float NdotH = __saturatef(dot(N, H));
    float D = alpha2 / (Pi * square(square(NdotH) * (alpha2 - 1.0f) + 1.0f));

    // F = Schlick-Fresnel
    float LdotH = __saturatef(dot(L, H));
    vector3 F = F0 + (vector3(1.0f) - F0) * pow5(1.0f - LdotH);

    float NdotL = __saturatef(dot(N, L));
    float NdotV = __saturatef(dot(N, V));

    // V = Schlick model with k chosen to better fit Smith model for GGX
    // No "hotness remapping"
    float k = alpha * 0.5f;
    // NdotL*NdotV is missing from this term
    float Vis = VisibilitySchlickSmith_G1V(NdotL, k) * VisibilitySchlickSmith_G1V(NdotV, k);
    // NdotL*NdotV is supposed to be in the denominator, cancel it with the missing terms from Vis
    return F * D * Vis / 4.0f;
}

// GGX_SchlickFresnel_SchlickSmithGGX with lambertian diffuse
CDI vector3
PBR_BRDF1(vector3 L, vector3 N, vector3 V, vector3 BaseColor, float roughness, vector3 F0, vector3 lightColor) {
    vector3 spec = GGX_SchlickFresnel_SchlickSmithGGX(N, L, V, roughness, F0);
    float NdotL = __saturatef(dot(N, L));
    vector3 diffuseContribution = BaseColor * lightColor * (vector3(1.0f) - spec) * NdotL;
    vector3 glossyContribution = lightColor * spec;
    return diffuseContribution + glossyContribution;
}

CDI vector3 PointLightContribution(
    vector3 P, vector3 N, vector3 V, vector3 DiffuseColor, float roughness, vector3 F0, LightPoint light) {
    vector3 Delta = light.Position - P;
    float r = length(Delta);
    vector3 L = normalize(Delta);
    float f_dist = __saturatef((light.FalloffEnd - r) * light.FalloffScale);
    vector3 lightColor = light.Color * f_dist;
    return PBR_BRDF1(L, N, V, DiffuseColor, roughness, F0, lightColor);
}

CDI vector3 DirectionalLightContribution(
    vector3 P, vector3 N, vector3 V, vector3 DiffuseColor, float roughness, vector3 F0, LightDirectional light) {
    return PBR_BRDF1(light.Direction, N, V, DiffuseColor, roughness, F0, light.Power);
}

CDI vector3 SpotLightContribution(
    vector3 P, vector3 N, vector3 V, vector3 DiffuseColor, float roughness, vector3 F0, LightSpot light) {
    vector3 Delta = light.Position - P;
    float r = length(Delta);
    vector3 L = normalize(Delta);
    float f_dist = __saturatef((light.FalloffEnd - r) * light.FalloffScale);
    vector3 lightColor = light.Color * f_dist;
    vector3 litColor = PBR_BRDF1(L, N, V, DiffuseColor, roughness, F0, lightColor);

    // Cone angle falloff
    float cos_a = dot(light.Direction, -L);
    float f_angle = __saturatef((cos_a - light.CosOuterAngle) * light.CosAngleScale);
    return litColor * f_angle;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// input should be pre-exposed, so input=1 produces output~=.8
// output is linear and will still need some sort of gamma treatment
// more comments on color spaces in the above link...
CDI vector3 ACESFilm(vector3 inputColor) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    vector3 result = ((inputColor * (a * inputColor + b)) / (inputColor * (c * inputColor + d) + e));
    return fastSaturate(result);
}

CDI vector4 ACESFilm(vector4 inputColor) {
    return vector4(ACESFilm(vector3(inputColor)), inputColor.w);
}

CHDI uint32_t ToColor4Unorm8(vector3 color) {
    uint32_t r = uint32_t(color.x * 255.0f);
    uint32_t g = uint32_t(color.y * 255.0f);
    uint32_t b = uint32_t(color.z * 255.0f);
    return (r) | (g << 8) | (b << 16) | (0xFF000000);
}
CHDI uint32_t ToColor4Unorm8SRgb(vector3 color) {
    color = linearToSRgb(color);
    uint32_t r = uint32_t(color.x * 255.0f);
    uint32_t g = uint32_t(color.y * 255.0f);
    uint32_t b = uint32_t(color.z * 255.0f);
    return (r) | (g << 8) | (b << 16) | (0xFF000000);
}

CHDI uint32_t ToColor4Unorm8(vector4 color) {
    uint32_t r = uint32_t(color.x * 255.0f);
    uint32_t g = uint32_t(color.y * 255.0f);
    uint32_t b = uint32_t(color.z * 255.0f);
    uint32_t a = uint32_t(color.w * 255.0f);
    return (r) | (g << 8) | (b << 16) | (a << 24);
}
CHDI uint32_t ToColor4Unorm8SRgb(vector4 color) {
    color = linearToSRgb(color);
    uint32_t r = uint32_t(color.x * 255.0f);
    uint32_t g = uint32_t(color.y * 255.0f);
    uint32_t b = uint32_t(color.z * 255.0f);
    uint32_t a = uint32_t(color.w * 255.0f);
    return (r) | (g << 8) | (b << 16) | (a << 24);
}
CHDI uint64_t ToColor4Unorm16(vector4 color) {
	const float C = 65535.0f; // 2^16-1
	uint64_t r = uint64_t(color.x * C);
	uint64_t g = uint64_t(color.y * C);
	uint64_t b = uint64_t(color.z * C);
	uint64_t a = uint64_t(color.w * C);
	return (r) | (g << 16) | (b << 32) | (a << 48);
}

CHDI vector4 FromColor4Unorm8(uint32_t c) {
    float r = (c & 0xFF) / 255.0f;
    float g = ((c >> 8) & 0xFF) / 255.0f;
    float b = ((c >> 16) & 0xFF) / 255.0f;
    float a = ((c >> 24) & 0xFF) / 255.0f;

    return vector4(r, g, b, a);
}
CHDI vector4 FromColor4Unorm8SRgb(uint32_t c) {
    float r = (c & 0xFF) / 255.0f;
    float g = ((c >> 8) & 0xFF) / 255.0f;
    float b = ((c >> 16) & 0xFF) / 255.0f;
    float a = ((c >> 24) & 0xFF) / 255.0f;

    return sRgbToLinear(vector4(r, g, b, a));
}

} // namespace hvvr
