/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "camera.h"
#include "console.h"
#include "debug.h"
#include "input.h"
#include "light.h"
#include "model_import.h"
#include "raycaster.h"
#include "raycaster_common.h"
#include "raycaster_spec.h"
#include "timer.h"
#include "vector_math.h"
#include "window_d3d11.h"

#include <ShellScalingAPI.h>
#include <stdio.h>

#pragma comment(lib, "Shcore.lib")

// disable camera movement for benchmarking?
#define DISABLE_MOVEMENT 0
#define CAMERA_SPEED 3.0

// 0 = off, 1 = match monitor refresh, 2 = half monitor refresh
#define ENABLE_VSYNC 0

#define ENABLE_DEPTH_OF_FIELD 0

// you might also want to enable JITTER_SAMPLES in kernel_constants.h
#define ENABLE_FOVEATED 0
// for foveated
#define GAZE_CURSOR_MODE_NONE 0 // eye direction is locked forward
#define GAZE_CURSOR_MODE_MOUSE 1 // eye direction is set by clicking the mouse on the window
#define GAZE_CURSOR_MODE GAZE_CURSOR_MODE_NONE
enum ModelviewerScene {
    scene_home = 0,
    scene_bunny,
    scene_conference,
    scene_sponza,
    scene_bistro_interior,
    SceneCount
};
// which scene to load?
static ModelviewerScene gSceneSelect = scene_sponza;

#define RT_WIDTH 2160
#define RT_HEIGHT 1200

static hvvr::Timer gTimer;

static std::unique_ptr<WindowD3D11> gWindow;
static std::unique_ptr<hvvr::Raycaster> gRayCaster;

static hvvr::Camera* gCamera = nullptr;
static hvvr::vector3 gCameraPos = {};
static float gCameraYaw = 0.0f;
static float gCameraPitch = 0.0f;

void gOnInit();
void gOnMain();
void gOnShutdown();

void resizeCallback() {
    uint32_t width = gWindow->getWidth();
    uint32_t height = gWindow->getHeight();

    hvvr::ImageViewR8G8B8A8 image((uint32_t*)gWindow->getRenderTargetTex(), width, height, width);
    hvvr::ImageResourceDescriptor renderTarget(image);
    renderTarget.memoryType = hvvr::ImageResourceDescriptor::MemoryType::DX_TEXTURE;

    hvvr::DynamicArray<hvvr::Sample> samples = hvvr::getGridSamples(width, height);

    gCamera->setViewport(hvvr::FloatRect{{-(float)width / height, -1}, {(float)width / height, 1}});
    gCamera->setRenderTarget(renderTarget);
    gCamera->setSamples(samples.data(), uint32_t(samples.size()), 1);
}

void rawMouseInputCallback(int dx, int dy) {
    (void)dx;
    (void)dy;
#if !DISABLE_MOVEMENT
    gCameraYaw += -dx * 0.001f;
    gCameraPitch += -dy * 0.001f;
#endif
}

int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, char* commandLine, int nCmdShow) {
    (void)hInstance;
    (void)hPrevInstance;
    (void)commandLine;
    (void)nCmdShow;

    // set the working directory to the executable's directory
    char exePath[MAX_PATH] = {};
    GetModuleFileName(GetModuleHandle(nullptr), exePath, MAX_PATH);
    char exeDir[MAX_PATH] = {};
    const char* dirTerminatorA = strrchr(exePath, '/');
    const char* dirTerminatorB = strrchr(exePath, '\\');
    const char* dirTerminator = hvvr::max(dirTerminatorA, dirTerminatorB);
    if (dirTerminator > exePath) {
        size_t dirLen = hvvr::min<size_t>(size_t(dirTerminator - exePath), MAX_PATH - 1);
        strncpy(exeDir, exePath, dirLen);
        SetCurrentDirectory(exeDir);
    }

    // disable scaling of the output window
    SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

    // create a console output window
    console::createStdOutErr();

    gWindow = std::make_unique<WindowD3D11>("Simpleviewer", RT_WIDTH, RT_HEIGHT, resizeCallback, rawMouseInputCallback);
    input::registerDefaultRawInputDevices(gWindow->getWindowHandle());

    gOnInit();

    // The main loop.
    MSG msg;
    for (;;) {
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                goto shutdown;
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        gOnMain();
    }

shutdown:
    gOnShutdown();

    return (int)msg.wParam;
}

void gOnInit() {
    RayCasterSpecification spec;
#if ENABLE_FOVEATED
    spec = RayCasterSpecification::feb2017FoveatedDemoSettings();
#else
    spec.mode = RayCasterSpecification::GPUMode::GPU_INTERSECT_AND_RECONSTRUCT_DEFERRED_MSAA_RESOLVE;
    spec.outputMode = RaycasterOutputMode::COLOR_RGBA8;
#endif
    gRayCaster = std::make_unique<hvvr::Raycaster>(spec);

    std::string sceneBasePath = "../../../../libraries/hvvr/samples_shared/data/scenes/";
    std::string scenePath;
    float sceneScale = 1.0f;
    switch (gSceneSelect) {
        case scene_home:
            // Oculus Home
            gCameraPos = hvvr::vector3(1.0f, 3.0f, -1.5f);
            gCameraYaw = -3.14159f * .7f;
            gCameraPitch = 3.14159f * -.05f;
            scenePath = sceneBasePath + "oculus_home.bin";
            break;

        case scene_bunny:
            // Stanford Bunny
            gCameraPos = hvvr::vector3(-0.253644f, 0.577575f, 1.081316f);
            gCameraYaw = -0.162111f;
            gCameraPitch = -0.453079f;
            scenePath = sceneBasePath + "bunny.bin";
            break;

        case scene_conference:
            // Conference Room
            gCameraPos = hvvr::vector3(10.091616f, 4.139270f, 1.230567f);
            gCameraYaw = -5.378105f;
            gCameraPitch = -0.398078f;
            scenePath = sceneBasePath + "conference.bin";
            break;

        case scene_sponza:
            // Crytek Sponza
            gCameraPos = hvvr::vector3(4.198845f, 6.105420f, -0.400903f);
            gCameraYaw = -4.704108f;
            gCameraPitch = -0.200078f;
            scenePath = sceneBasePath + "sponza.bin";
            sceneScale = .01f;
            break;

        case scene_bistro_interior:
            // Amazon Bistro
            gCameraPos = hvvr::vector3(2.0f, 2.0f, -0.5f);
            gCameraYaw = -3.14159f * .5f;
            gCameraPitch = 3.14159f * -.05f;
            scenePath = sceneBasePath + "bistro.bin";
            break;

        default:
            hvvr::fail("invalid scene enum");
            return;
            break;
    }

    // add a default directional light
    hvvr::LightUnion light = {};
    light.type = hvvr::LightType::directional;
    light.directional.Direction = hvvr::normalize(hvvr::vector3(-.25f, 1.0f, 0.1f));
    light.directional.Power = hvvr::vector3(0.4f, 0.35f, 0.35f);
    gRayCaster->createLight(light);

#if ENABLE_DEPTH_OF_FIELD
    const float lensRadius = 0.0015f; // avg 3mm diameter in light (narrow pupil)
#else
    const float lensRadius = 0.0f;
#endif
    const float focalDistance = .1f; // min focal dist is about .1m for average gamer
    gCamera = gRayCaster->createCamera(hvvr::FloatRect(hvvr::vector2(-1, -1), hvvr::vector2(1, 1)), lensRadius);
    gCamera->setFocalDepth(focalDistance);
    resizeCallback(); // make sure we bind a render target and some samples to the camera

    // load the scene
    model_import::Model importedModel;
    if (!model_import::load(scenePath.c_str(), importedModel)) {
        hvvr::fail("failed to load model %s", scenePath.c_str());
    }

    // apply scaling
    for (auto& mesh : importedModel.meshes) {
        mesh.transform.scale *= sceneScale;
    }
    // create the scene objects in the raycaster
    if (!model_import::createObjects(*gRayCaster, importedModel)) {
        hvvr::fail("failed to create model objects");
    }
}

void gOnShutdown() {
    gCamera = nullptr;
    gRayCaster = nullptr;
}

void gOnMain() {
    static uint64_t frameID = 0;
    static double prevElapsedTime = gTimer.getElapsed();
    double elapsedTime = gTimer.getElapsed();
    float deltaTime = float(elapsedTime - prevElapsedTime);
    prevElapsedTime = elapsedTime;

    hvvr::vector3 posDelta(0.0f);
#if !DISABLE_MOVEMENT
    float cardinalCameraSpeed = (float)(CAMERA_SPEED * deltaTime);
    if (GetAsyncKeyState(VK_LSHIFT) & 0x8000)
        cardinalCameraSpeed *= .05f;

    if (GetAsyncKeyState('W') & 0x8000)
        posDelta.z -= cardinalCameraSpeed;
    if (GetAsyncKeyState('A') & 0x8000)
        posDelta.x -= cardinalCameraSpeed;
    if (GetAsyncKeyState('S') & 0x8000)
        posDelta.z += cardinalCameraSpeed;
    if (GetAsyncKeyState('D') & 0x8000)
        posDelta.x += cardinalCameraSpeed;
    if (GetAsyncKeyState(VK_LCONTROL) & 0x8000)
        posDelta.y -= cardinalCameraSpeed;
    if (GetAsyncKeyState(VK_SPACE) & 0x8000)
        posDelta.y += cardinalCameraSpeed;
    gCameraPos += hvvr::matrix3x3(hvvr::quaternion::fromEulerAngles(gCameraYaw, gCameraPitch, 0)) * posDelta;
#endif

#if GAZE_CURSOR_MODE == GAZE_CURSOR_MODE_MOUSE
    if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) {
        POINT cursorCoord = {};
        GetCursorPos(&cursorCoord);
        ScreenToClient(HWND(gWindow->getWindowHandle()), &cursorCoord);

        RECT clientRect = {};
        GetClientRect(HWND(gWindow->getWindowHandle()), &clientRect);
        int width = clientRect.right - clientRect.left;
        int height = clientRect.bottom - clientRect.top;

        float cursorX = float(cursorCoord.x) / width * 2.0f - 1.0f;
        float cursorY = -(float(cursorCoord.y) / height * 2.0f - 1.0f);

        if (fabsf(cursorX) <= 1.0f && fabsf(cursorY) <= 1.0f) {
            float screenDistance = 1.0f;

            // This doesn't attempt to take into account the actual field of view or mapping from samples to pixels,
            // so it will be a bit off, especially as you get further from the screen center.
            hvvr::vector3 cursorPosEye(cursorX, cursorY, -screenDistance);
            hvvr::vector3 eyeDir = hvvr::normalize(cursorPosEye);
            gCamera->setEyeDir(eyeDir);
        }
    }
#endif

    hvvr::transform worldFromCamera =
        hvvr::transform(gCameraPos, hvvr::quaternion::fromEulerAngles(gCameraYaw, gCameraPitch, 0), 1.0f);
    gCamera->setCameraToWorld(worldFromCamera);

    gRayCaster->render(elapsedTime);

#if OUTPUT_MODE == OUTPUT_MODE_3D_API
    uint32_t syncInterval = ENABLE_VSYNC;
    gWindow->copyAndPresent(syncInterval);
#endif

    // collect some overall perf statistics
    {
        struct FrameStats {
            float deltaTime;
            uint32_t rayCount;
        };
        enum { frameStatsWindowSize = 64 };
        static FrameStats frameStats[frameStatsWindowSize] = {};
        static int frameStatsPos = 0;

        uint32_t rayCount = gCamera->getSampleData().sampleCount;

        frameStats[frameStatsPos].deltaTime = deltaTime;
        frameStats[frameStatsPos].rayCount = rayCount;

        // let it run for a bit before collecting numbers
        if (frameStatsPos == 0 && frameID > frameStatsWindowSize * 4) {
            static double frameTimeAvgTotal = 0.0;
            static uint64_t frameTimeAvgCount = 0;

            // search for the fastest frame in the history window which matches the current state of the raycaster
            int fastestMatchingFrame = -1;
            for (int n = 0; n < frameStatsWindowSize; n++) {
                frameTimeAvgTotal += frameStats[n].deltaTime;
                frameTimeAvgCount++;

                if (fastestMatchingFrame == -1 ||
                    frameStats[n].deltaTime < frameStats[fastestMatchingFrame].deltaTime) {
                    fastestMatchingFrame = n;
                }
            }
            assert(fastestMatchingFrame >= 0 && fastestMatchingFrame < frameStatsWindowSize);
            const FrameStats& fastestFrame = frameStats[fastestMatchingFrame];

            float frameTimeAvg = float(frameTimeAvgTotal / double(frameTimeAvgCount));

            printf("%.0f (%.0f) mrays/s"
                   ", %.2f (%.2f) ms frametime"
                   ", %u x %u rays"
                   "\n",
                   fastestFrame.rayCount / fastestFrame.deltaTime / 1000000.0f * hvvr::COLOR_MODE_MSAA_RATE,
                   fastestFrame.rayCount / frameTimeAvg / 1000000.0f * hvvr::COLOR_MODE_MSAA_RATE,
                   fastestFrame.deltaTime * 1000, frameTimeAvg * 1000, fastestFrame.rayCount,
                   hvvr::COLOR_MODE_MSAA_RATE);
        }
        frameStatsPos = (frameStatsPos + 1) % frameStatsWindowSize;
    }

    frameID++;
}
