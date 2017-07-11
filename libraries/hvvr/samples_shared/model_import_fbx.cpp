/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model_import.h"

#if MODEL_IMPORT_ENABLE_FBX

#include "bvh.h"
#include "constants_math.h"
#include "graphics_types.h"
#include "light.h"
#include "material.h"
#include "model.h"
#include "texture.h"
#include "util_graphics.h"

#pragma warning(push)
#pragma warning(disable : 4244 4456)
#define STB_IMAGE_IMPLEMENTATION
#include "3rdparty/stb_image.h"
#pragma warning(pop)

#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include <fbxsdk.h>
#ifdef _DEBUG
# pragma comment(lib, "debug/libfbxsdk-md.lib")
#else
# pragma comment(lib, "release/libfbxsdk-md.lib")
#endif

namespace {

typedef std::unordered_map<std::string, uint32_t> TextureLookupType;
struct ImportState {
    ImportState(model_import::Model& _model) : model(_model) {}

    model_import::Model& model;
    TextureLookupType textureLookup;
};

bool importNode(ImportState& state, FbxNode* pNode);
bool importMesh(ImportState& state, FbxNode* pNode);
bool importMaterial(ImportState& state, model_import::Mesh& mesh, FbxSurfaceMaterial* pFbxMaterial);
bool importLight(ImportState& state, FbxNode* pNode, FbxLight* pLight);
uint32_t importTexture(ImportState& state, FbxFileTexture* pFileTexture, bool sRGB);

bool importFbx(const char* path, model_import::Model& model) {
    auto managerDeleter = [](FbxManager* manager) { manager->Destroy(); };
    std::unique_ptr<FbxManager, decltype(managerDeleter)> fbxManager(FbxManager::Create(), managerDeleter);

    FbxIOSettings* ioSettings = FbxIOSettings::Create(fbxManager.get(), IOSROOT);
    FbxScene* scene = FbxScene::Create(fbxManager.get(), "ImportScene");
    FbxImporter* importer = FbxImporter::Create(fbxManager.get(), "");
    fbxManager->SetIOSettings(ioSettings);
    printf("info: Autodesk FBX SDK version %s\n", fbxManager->GetVersion());

    if (!importer->Initialize(path, -1, ioSettings)) {
        FbxString error = importer->GetStatus().GetErrorString();
        printf("error: Call to FbxImporter::Initialize() failed.\n");
        printf("error: Error returned: %s\n", error.Buffer());

        if (importer->GetStatus().GetCode() == FbxStatus::eInvalidFileVersion) {
            int lFileMajor, lFileMinor, lFileRevision;
            int lSDKMajor, lSDKMinor, lSDKRevision;
            // Get the file version number generate by the FBX SDK.
            FbxManager::GetFileFormatVersion(lSDKMajor, lSDKMinor, lSDKRevision);
            importer->GetFileVersion(lFileMajor, lFileMinor, lFileRevision);
            printf("error: FBX file format version for this FBX SDK is %d.%d.%d\n", lSDKMajor, lSDKMinor, lSDKRevision);
            printf("error: FBX file format version for file '%s' is %d.%d.%d\n", path, lFileMajor, lFileMinor,
                   lFileRevision);
        }

        printf("error: Failed to initialize the importer for '%s'.\n", path);
        return false;
    }

    // Set the import states. Default is true.
    ioSettings->SetBoolProp(IMP_FBX_MATERIAL, true);
    ioSettings->SetBoolProp(IMP_FBX_TEXTURE, true);
    ioSettings->SetBoolProp(IMP_FBX_LINK, true);
    ioSettings->SetBoolProp(IMP_FBX_SHAPE, true);
    ioSettings->SetBoolProp(IMP_FBX_GOBO, true);
    ioSettings->SetBoolProp(IMP_FBX_ANIMATION, false);
    ioSettings->SetBoolProp(IMP_FBX_GLOBAL_SETTINGS, true);

    // Import the scene.
    if (!importer->Import(scene)) {
        if (importer->GetStatus().GetCode() == FbxStatus::ePasswordError) {
            printf("error: File is password protected.\n");
            return false;
        }

        printf("error: Failed to load the scene : '%s'\n", path);
        return false;
    }

    // convert to meters
    if (scene->GetGlobalSettings().GetSystemUnit() != FbxSystemUnit::m)
        FbxSystemUnit::m.ConvertScene(scene);

    // triangulate
    FbxGeometryConverter GeometryConverter(fbxManager.get());
    GeometryConverter.Triangulate(scene, true, false);

    FbxNode* root = scene->GetRootNode();
    if (!root) {
        printf("error: Scene has no root node when importing '%s'.\n", path);
        return false;
    }

    ImportState state(model);
    if (!importNode(state, root))
        return false;
    return true;
}

bool importNode(ImportState& state, FbxNode* pNode) {
    if (auto pLight = pNode->GetLight()) {
        if (!importLight(state, pNode, pLight))
            return false;
    } else if (auto pMesh = pNode->GetMesh()) {
        if (!importMesh(state, pNode))
            return false;
    }

    for (int i = 0, e = pNode->GetChildCount(); i < e; i++) {
        if (!importNode(state, pNode->GetChild(i))) {
            return false;
        }
    }

    return true;
}

hvvr::transform AsTransform(const FbxAMatrix& a) {
    FbxQuaternion q = a.GetQ();
    FbxVector4 t = a.GetT();
    FbxVector4 s = a.GetS();
    return hvvr::transform(hvvr::vector3(float(t[0]), float(t[1]), float(t[2])),
                           hvvr::quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])), float(s[0]));
}

hvvr::transform GetGeometryTransform(FbxNode* pNode) {
    auto lT = pNode->GetGeometricTranslation(FbxNode::eSourcePivot);
    auto lR = pNode->GetGeometricRotation(FbxNode::eSourcePivot);
    auto lS = pNode->GetGeometricScaling(FbxNode::eSourcePivot);

    auto lM = FbxAMatrix(lT, lR, lS);
    auto lQ = lM.GetQ();
    return hvvr::transform(hvvr::vector3(float(lT[0]), float(lT[1]), float(lT[2])),
                           hvvr::quaternion(float(lQ[0]), float(lQ[1]), float(lQ[2]), float(lQ[3])), float(lS[0]));
}

// Get the value of a geometry element for a triangle vertex
template <typename TGeometryElement, typename TValue>
TValue GetVertexElement(TGeometryElement* pElement, int iPoint, int iTriangle, int iVertex, TValue defaultValue) {
    if (!pElement || pElement->GetMappingMode() == FbxGeometryElement::eNone)
        return defaultValue;
    int index = 0;

    if (pElement->GetMappingMode() == FbxGeometryElement::eByControlPoint)
        index = iPoint;
    else if (pElement->GetMappingMode() == FbxGeometryElement::eByPolygon)
        index = iTriangle;
    else if (pElement->GetMappingMode() == FbxGeometryElement::eByPolygonVertex)
        index = iTriangle * 3 + iVertex;

    if (pElement->GetReferenceMode() != FbxGeometryElement::eDirect)
        index = pElement->GetIndexArray().GetAt(index);

    return pElement->GetDirectArray().GetAt(index);
}

template <typename KeyType, typename ValueType>
struct UnorderedMapGenerator {
    struct Hash {
        uint32_t operator()(const KeyType& a) const {
            uint32_t digest = 0;
            for (size_t i = 0; i < sizeof(a); i++)
                digest = _mm_crc32_u8(digest, ((uint8_t*)&a)[i]);
            return digest;
        }
    };
    struct CompareEq {
        bool operator()(const KeyType& a, const KeyType& b) const {
            return !memcmp(&a, &b, sizeof(a));
        }
    };
    typedef std::unordered_map<KeyType, ValueType, Hash, CompareEq> Type;
};

bool importMesh(ImportState& state, FbxNode* pNode) {
    auto pMesh = pNode->GetMesh();
    if (!pMesh->IsTriangleMesh()) {
        printf("error: We only support triangle meshes.\n");
        return false;
    }

    model_import::Mesh mesh;
    mesh.transform = AsTransform(pNode->EvaluateGlobalTransform()) * GetGeometryTransform(pNode);

    // Import the materials.
    int materialCount = pNode->GetMaterialCount();
    for (int n = 0; n < materialCount; n++) {
        if (!importMaterial(state, mesh, pNode->GetMaterial(n))) {
            return false;
        }
    }

    const FbxGeometryElementNormal* pNormals = pMesh->GetElementNormal(0);
    if (!pNormals) {
        // Generate normals if we don't have any
        pMesh->GenerateNormals();
        pNormals = pMesh->GetElementNormal(0);
    }
    const FbxGeometryElementUV* pUVs = pMesh->GetElementUV(0);
    const FbxLayerElementMaterial* pPolygonMaterials = pMesh->GetElementMaterial();
    assert(pPolygonMaterials != nullptr);
    const auto& pPolygonMaterialsIndexArray = pPolygonMaterials->GetIndexArray();

    // vertex deduplication
    UnorderedMapGenerator<hvvr::ShadingVertex, uint32_t>::Type hashMap;

    uint32_t numTriangles = uint32_t(pMesh->GetPolygonCount());
    mesh.data.verts.reserve(numTriangles * 3);
    mesh.data.triShade.resize(numTriangles);

    for (uint32_t t = 0; t < numTriangles; t++) {
        uint32_t triIndices[3];
        for (uint32_t v = 0; v < 3; v++) {
            int iPoint = pMesh->GetPolygonVertex(t, v);

            FbxVector4 point = pMesh->GetControlPointAt(iPoint);
            FbxVector4 normal = GetVertexElement(pNormals, iPoint, t, v, FbxVector4(0, 0, 0, 0));
            FbxVector2 uv = GetVertexElement(pUVs, iPoint, t, v, FbxVector2(0, 1));

            hvvr::ShadingVertex vertex = {};
            vertex.pos = hvvr::vector3(float(point[0]), float(point[1]), float(point[2]));
            vertex.normal = hvvr::vector4h(hvvr::vector4(float(normal[0]), float(normal[1]), float(normal[2]), 0));
            vertex.uv = hvvr::vector2(float(uv[0]), 1.0f - float(uv[1]));

            auto it = hashMap.find(vertex);
            if (it != hashMap.end()) {
                // it's a duplicate vertex
                triIndices[v] = it->second;
            } else {
                // we haven't run into this vertex yet
                uint32_t index = uint32_t(mesh.data.verts.size());
                mesh.data.verts.emplace_back(vertex);
                hashMap[vertex] = index;
                triIndices[v] = index;
            }
        }

        int materialIndex = pPolygonMaterialsIndexArray.GetAt(t);
        assert(materialIndex >= 0 && materialIndex < materialCount);

        hvvr::PrecomputedTriangleShade& triShade = mesh.data.triShade[t];
        triShade.indices[0] = triIndices[0];
        triShade.indices[1] = triIndices[1];
        triShade.indices[2] = triIndices[2];
        triShade.material = uint32_t(materialIndex);
    }

    hvvr::GenerateTopology(mesh.data);

    state.model.meshes.emplace_back(std::move(mesh));
    return true;
}

FbxFileTexture* GetTextureFromProperty(FbxProperty* pProp) {
    return (FbxFileTexture*)pProp->GetSrcObject(FbxCriteria::ObjectType(FbxFileTexture::ClassId));
}

// convert from sRGB to linear, and pre-multiply
hvvr::vector4 fbxColorConvert(const FbxDouble3& inColor, const FbxDouble& inFactor) {
    hvvr::vector3 linearColor = hvvr::sRgbToLinear(hvvr::vector3(float(inColor[0]), float(inColor[1]), float(inColor[2])));
    return hvvr::vector4(linearColor * float(inFactor), float(inFactor));
}

// Import a material
bool importMaterial(ImportState& state, model_import::Mesh& mesh, FbxSurfaceMaterial* pFbxMaterial) {
    if (pFbxMaterial == nullptr) {
        printf("error: Material was null!\n");
        return false;
    }

    hvvr::SimpleMaterial material = {};
    if (pFbxMaterial->GetClassId().Is(FbxSurfacePhong::ClassId)) {
        FbxSurfacePhong* pPhong = (FbxSurfacePhong*)pFbxMaterial;

        FbxDouble3 Diffuse = pPhong->Diffuse.Get();
        FbxDouble DiffuseFactor = pPhong->DiffuseFactor.Get();
        FbxDouble3 Specular = pPhong->Specular.Get();
        FbxDouble SpecularFactor = pPhong->SpecularFactor.Get();
        FbxDouble3 Emissive = pPhong->Emissive.Get();
        FbxDouble EmissiveFactor = pPhong->EmissiveFactor.Get();
        FbxDouble3 TransColor = pPhong->TransparentColor.Get();
        FbxDouble TransFactor = pPhong->TransparencyFactor.Get();
        FbxDouble Shininess = pPhong->Shininess.Get();

        float transparency = float((TransColor[0] + TransColor[1] + TransColor[2]) / 3.0 * TransFactor);
        // Undo FBX glossiness to specular power mapping (n = 2^(g*10)) to get a 0..1 gloss value
        float glossiness = float(log2(fmax(Shininess, 1.0)) / 10.0);

        uint32_t diffuseID = importTexture(state, GetTextureFromProperty(&pPhong->Diffuse), true);
        uint32_t emissiveID = importTexture(state, GetTextureFromProperty(&pPhong->Emissive), true);
        uint32_t specularID = importTexture(state, GetTextureFromProperty(&pPhong->Specular), true);
        uint32_t glossinessID = importTexture(state, GetTextureFromProperty(&pPhong->Shininess), true);

        material.emissive = fbxColorConvert(Emissive, EmissiveFactor);
        material.diffuse = fbxColorConvert(Diffuse, DiffuseFactor);
        material.specular = fbxColorConvert(Specular, SpecularFactor);
        material.glossiness = glossiness;
        material.opacity = 1.0f - transparency;

        if (emissiveID != hvvr::SimpleMaterial::badTextureIndex) {
            material.textureIDsAndShadingModel = hvvr::SimpleMaterial::buildShadingCode(
                hvvr::ShadingModel::emissive, emissiveID, hvvr::SimpleMaterial::badTextureIndex,
                hvvr::SimpleMaterial::badTextureIndex);
        } else {
            material.textureIDsAndShadingModel =
                hvvr::SimpleMaterial::buildShadingCode(hvvr::ShadingModel::phong, diffuseID, specularID, glossinessID);
        }
    } else if (pFbxMaterial->GetClassId().Is(FbxSurfaceLambert::ClassId)) {
        FbxSurfaceLambert* pLam = (FbxSurfaceLambert*)pFbxMaterial;

        FbxDouble3 Diffuse = pLam->Diffuse.Get();
        FbxDouble DiffuseFactor = pLam->DiffuseFactor.Get();
        FbxDouble3 Emissive = pLam->Emissive.Get();
        FbxDouble EmissiveFactor = pLam->EmissiveFactor.Get();
        FbxDouble3 TransColor = pLam->TransparentColor.Get();
        FbxDouble TransFactor = pLam->TransparencyFactor.Get();

        float transparency = 1.0f - float((TransColor[0] + TransColor[1] + TransColor[2]) / 3.0 * TransFactor);

        uint32_t diffuseID = importTexture(state, GetTextureFromProperty(&pLam->Diffuse), true);
        uint32_t emissiveID = importTexture(state, GetTextureFromProperty(&pLam->Emissive), true);
        uint32_t specularID = hvvr::SimpleMaterial::badTextureIndex;
        uint32_t glossinessID = hvvr::SimpleMaterial::badTextureIndex;

        material.emissive = fbxColorConvert(Emissive, EmissiveFactor);
        material.diffuse = fbxColorConvert(Diffuse, DiffuseFactor);
        material.specular = hvvr::vector4(0, 0, 0, 0);
        material.glossiness = 1.0f;
        material.opacity = 1.0f - transparency;

        if (emissiveID != hvvr::SimpleMaterial::badTextureIndex) {
            material.textureIDsAndShadingModel = hvvr::SimpleMaterial::buildShadingCode(
                hvvr::ShadingModel::emissive, emissiveID, hvvr::SimpleMaterial::badTextureIndex,
                hvvr::SimpleMaterial::badTextureIndex);
        } else {
            material.textureIDsAndShadingModel =
                hvvr::SimpleMaterial::buildShadingCode(hvvr::ShadingModel::phong, diffuseID, specularID, glossinessID);
        }
    } else {
        printf("error: Unknown material type!\n");
        return false;
    }

    mesh.data.materials.emplace_back(std::move(material));
    return true;
}

bool importLight(ImportState& state, FbxNode* pNode, FbxLight* pLight) {
    hvvr::transform cframe = AsTransform(pNode->EvaluateGlobalTransform());

    FbxDouble3 Color = pLight->Color.Get();
    hvvr::vector3 color = hvvr::vector3(float(Color[0]), float(Color[1]), float(Color[2]));
    color = hvvr::sRgbToLinear(color);

    float intensity = float(pLight->Intensity.Get() * 0.01);

    // TODO: is this necessary? It was needed for Oculus Home...
    float falloffStart = float(pLight->FarAttenuationStart.Get()) * 0.01f;
    float falloffEnd = float(pLight->FarAttenuationEnd.Get()) * 0.01f;

    hvvr::LightUnion light;
    if (pLight->LightType.Get() == FbxLight::eDirectional) {
        light.type = hvvr::LightType::directional;
        light.directional.Direction = hvvr::normalize(cframe * hvvr::vector3(0, 0, -1));
        light.directional.Power = color * intensity;
    } else if (pLight->LightType.Get() == FbxLight::ePoint) {
        if (bool(pLight->EnableFarAttenuation) == false) {
            printf("\nIMPORT WARNING: Ignoring point light because FarAttenuation not enabled.\n");
            return true; // we won't consider this a failure case
        }

        light.type = hvvr::LightType::point;
        light.point.Color = color;
        light.point.Position = cframe.translation;
        light.point.FalloffEnd = falloffEnd;
        light.point.FalloffScale = 1.0f / (falloffEnd - falloffStart);
    } else if (pLight->LightType.Get() == FbxLight::eSpot) {
        if (bool(pLight->EnableFarAttenuation) == false) {
            printf("\nIMPORT WARNING: Ignoring spot light because FarAttenuation not enabled.\n");
            return true; // we won't consider this a failure case
        }

        float cosInnerAngle = cosf(float(pLight->InnerAngle.Get()) * hvvr::RadiansPerDegree);
        float cosOuterAngle = cosf(float(pLight->OuterAngle.Get()) * hvvr::RadiansPerDegree);

        light.type = hvvr::LightType::spot;
        light.spot.Direction = hvvr::normalize(cframe * hvvr::vector3(0, 0, -1));
        light.spot.Color = color;
        light.spot.Position = cframe.translation;
        light.spot.FalloffEnd = falloffEnd;
        light.spot.FalloffScale = 1.0f / (falloffEnd - falloffStart);
        light.spot.CosOuterAngle = cosOuterAngle;
        light.spot.CosAngleScale = 1.0f / (cosInnerAngle - cosOuterAngle);
    }

    state.model.lights.emplace_back(std::move(light));
    return true;
}

uint32_t importTexture(ImportState& state, FbxFileTexture* pFileTexture, bool sRGB) {
    if (!pFileTexture)
        return hvvr::SimpleMaterial::badTextureIndex;

    const char* path = pFileTexture->GetFileName();
    auto it = state.textureLookup.find(path);
    if (it != state.textureLookup.end()) {
        // texture has already been loaded
        return it->second;
    }

    assert(state.model.textures.size() < hvvr::SimpleMaterial::badTextureIndex);

    int channelsOut = 4;
    int width, height, channelsIn;
    uint8_t* data = stbi_load(path, &width, &height, &channelsIn, channelsOut);
    if (data == nullptr) {
        printf("error: stb_image couldn't load %s\n", path);
        return hvvr::SimpleMaterial::badTextureIndex;
    }

    size_t size = width * height * channelsOut;

    hvvr::TextureData tex;
    tex.data = new uint8_t[size];
    memcpy((void*)tex.data, data, size);
    STBI_FREE(data);
    tex.format = sRGB ? hvvr::TextureFormat::r8g8b8a8_unorm_srgb : hvvr::TextureFormat::r8g8b8a8_unorm;
    tex.width = width;
    tex.height = height;
    tex.stride = width;

    uint32_t texIndex = uint32_t(state.model.textures.size());
    state.model.textures.emplace_back(std::move(tex));
    state.textureLookup[path] = texIndex;

    return texIndex;
}
} // namespace

namespace model_import {

bool loadFbx(const char* path, Model& model) {
    if (!importFbx(path, model)) {
        return false;
    }

    return true;
}

} // namespace model_import

#endif // MODEL_IMPORT_ENABLE_FBX
