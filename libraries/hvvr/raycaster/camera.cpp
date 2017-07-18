/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "camera.h"
#include "gpu_camera.h"
#include "gpu_context.h"
#include "sample_clustering.h"
#include "vector_math.h"

#include <functional>
#include <random>


namespace hvvr {

SampleData::SampleData(const Sample* rawSamples,
                       uint32_t rawSampleCount,
                       uint32_t splitColorSamples,
                       const matrix3x3& sampleToCamera,
                       ThinLens lens,
                       uint32_t rtWidth,
                       uint32_t rtHeight)
    : splitColorSamples(splitColorSamples), lens(lens) {
    DynamicArray<SortedSample> sortedSamples(rawSampleCount);
    for (size_t n = 0; n < rawSampleCount; n++) {
        sortedSamples[n] = SortedSample(rawSamples[n], n % splitColorSamples);
    }
    uint32_t blockCount = uint32_t((sortedSamples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // TODO(whunt): allow different clustering methods
    naiveXYCluster(ArrayView<SortedSample>(sortedSamples), blockCount);

    sampleBounds = {vector2(1.0f, 1.0f), vector2(0.0f, 0.0f)};
    for (const auto& s : sortedSamples) {
        sampleBounds.lower.x = std::min(sampleBounds.lower.x, s.position.x);
        sampleBounds.lower.y = std::min(sampleBounds.lower.y, s.position.y);
        sampleBounds.upper.x = std::max(sampleBounds.upper.x, s.position.x);
        sampleBounds.upper.y = std::max(sampleBounds.upper.y, s.position.y);
    }
    FloatRect cullRect;
    cullRect.lower.x = -INFINITY;
    cullRect.lower.y = -INFINITY;
    cullRect.upper.x = INFINITY;
    cullRect.upper.y = INFINITY;
    validSampleCount = uint32_t(rawSampleCount);
    samples.blockFrusta2D = DynamicArray<RayPacketFrustum2D>(blockCount);
    samples.tileFrusta2D = DynamicArray<RayPacketFrustum2D>(blockCount * TILES_PER_BLOCK);
    samples.blockFrusta3D = DynamicArray<RayPacketFrustum3D>(blockCount);
    samples.tileFrusta3D = DynamicArray<RayPacketFrustum3D>(blockCount * TILES_PER_BLOCK);
    blockedSamplePositions = DynamicArray<float>(blockCount * BLOCK_SIZE * 2);
    blockedSampleExtents = DynamicArray<Sample::Extents>(blockCount * BLOCK_SIZE);
    samples.generate(sortedSamples, blockCount, validSampleCount, cullRect, blockedSamplePositions,
                     blockedSampleExtents, lens, sampleToCamera);
    sampleCount = uint32_t(blockCount * BLOCK_SIZE);

    imageLocationToSampleIndex = DynamicArray<int32_t>(rtWidth * rtHeight * splitColorSamples);
    memset(imageLocationToSampleIndex.data(), 0xff, sizeof(int32_t) * imageLocationToSampleIndex.size()); // clear to -1
    for (size_t i = 0; i < validSampleCount; ++i) {
        vector2ui loc = sortedSamples[i].pixelLocation;
        uint32_t channel = sortedSamples[i].channel;
        imageLocationToSampleIndex[(loc.y * rtWidth + loc.x) * splitColorSamples + channel] = int32_t(i);
    }
}

Camera::Camera(const FloatRect& viewport, float apertureRadius, GPUContext& gpuContext)
    : _gpuCamera(nullptr), _lens({apertureRadius, 1.0f}), _eyeDir(0.0f, 0.0f, -1.0f) {
    setViewport(viewport);

    std::uniform_real_distribution<float> uniformRandomDist(0.0f, 1.0f);
    std::mt19937 generator;
    auto rand = std::bind(uniformRandomDist, std::ref(generator));
    for (int i = 0; i < 16; ++i) {
        _frameJitters.push_back({rand(), rand()});
    }

    bool created = false;
    _gpuCamera = &gpuContext.getCreateCamera(this, created);
    assert(created == true);
}

Camera::~Camera() {}

void Camera::setEnabled(bool enabled) {
    _enabled = enabled;
}

bool Camera::getEnabled() const {
    return _enabled;
}

void Camera::setEyeDir(const vector3& eyeDir) {
    _eyeDir = eyeDir;
}

const vector3& Camera::getEyeDir() const {
    return _eyeDir;
}

void Camera::setViewport(const FloatRect& viewport) {
    _viewport = viewport;
}

const FloatRect& Camera::getViewport() const {
    return _viewport;
}

void Camera::setLens(const ThinLens& lens) {
    _lens = lens;
}

void Camera::setAperture(float radius) {
    _lens.radius = radius;
}

float Camera::getAperture() const {
    return _lens.radius;
}

void Camera::setFocalDepth(float depth) {
    _lens.focalDistance = depth;
}

void Camera::setRenderTarget(const ImageResourceDescriptor& newRenderTarget) {
    if (newRenderTarget.isHardwareRenderTarget() && newRenderTarget != _renderTarget) {
        _newHardwareTarget = true;
    }
    _renderTarget = newRenderTarget;
}

void Camera::setSamples(const Sample* rawSamples, uint32_t rawSampleCount, uint32_t splitColorSamples) {
    setSampleData(SampleData(rawSamples, rawSampleCount, splitColorSamples, getSampleToCamera(), _lens,
                             _renderTarget.width, _renderTarget.height));
}

void Camera::setSampleData(const SampleData& sampleData) {
    _sampleData = sampleData;

    uint32_t blockCount = uint32_t(_sampleData.samples.blockFrusta3D.size());
    uint32_t tileCount = uint32_t(_sampleData.samples.tileFrusta3D.size());

    if (blockCount != _blockFrustaTransformed.size()) {
        _blockFrustaTransformed = DynamicArray<RayPacketFrustum3D>(blockCount);
    }
    if (tileCount != _tileFrustaTransformed.size()) {
        _tileFrustaTransformed = DynamicArray<RayPacketFrustum3D>(tileCount);
    }

    _gpuCamera->updateConfig(_outputMode, sampleData.imageLocationToSampleIndex.data(),
                             sampleData.blockedSamplePositions.data(), sampleData.blockedSampleExtents.data(), _lens,
                             sampleData.sampleCount, _renderTarget.width, _renderTarget.height,
                             uint32_t(_renderTarget.stride), sampleData.splitColorSamples);
}

const SampleData& Camera::getSampleData() const {
    return _sampleData;
}

matrix3x3 Camera::getSampleToCamera() const {
    return matrix3x3(vector3(_viewport.upper.x - _viewport.lower.x, 0, 0),
                     vector3(0, _viewport.lower.y - _viewport.upper.y, 0),
                     vector3(_viewport.lower.x, _viewport.upper.y, -1));
}

matrix4x4 Camera::getSampleToWorld() const {
    return matrix4x4(_cameraToWorld) * matrix4x4(getSampleToCamera());
}

matrix4x4 Camera::getWorldToSample() const {
    return invert(getSampleToWorld());
}

void Camera::setCameraToWorld(const transform& cameraToWorld) {
    _cameraToWorld = cameraToWorld;
}

matrix4x4 Camera::getCameraToWorld() const {
    return matrix4x4(_cameraToWorld);
}

const vector3& Camera::getTranslation() const {
    return _cameraToWorld.translation;
}

vector3 Camera::getForward() const {
    return vector3(-normalize(getCameraToWorld().m2));
}

} // namespace hvvr
