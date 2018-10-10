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
#include <fstream>
#include <sstream>
#include <stdio.h>

#pragma comment(lib, "Shcore.lib")


#define RT_WIDTH 2160
#define RT_HEIGHT 1200

// for foveated
#define GAZE_CURSOR_MODE_NONE 0  // eye direction is locked forward
#define GAZE_CURSOR_MODE_MOUSE 1 // eye direction is set by clicking the mouse on the window
#define GAZE_CURSOR_MODE GAZE_CURSOR_MODE_NONE

// disable camera movement (for benchmarking)
#define DISABLE_MOVEMENT 0
#define CAMERA_SPEED 3.0

// 0 = off, 1 = match monitor refresh, 2 = half monitor refresh
#define ENABLE_VSYNC 0

#define ENABLE_DEPTH_OF_FIELD 0

// you might also want to enable JITTER_SAMPLES in kernel_constants.h
#define ENABLE_FOVEATED 0

#define ENABLE_WIDE_FOV 0


enum ModelviewerScene {
    scene_home = 0,
    scene_bunny,
    scene_conference,
    scene_sponza,
    scene_bistro_interior,
    SceneCount
};
// which scene to load? Can be overwritten in the command line
static ModelviewerScene gSceneSelect = scene_sponza;

using hvvr::vector3;
struct SceneSpecification {
    vector3 defaultCameraOrigin;
    float defaultCameraYaw;
    float defaultCameraPitch;
    float scale;
    std::string filename;
};
static SceneSpecification gSceneSpecs[ModelviewerScene::SceneCount] = {
    {vector3(1.0f, 3.0f, -1.5f), -(float)M_PI * .7f, (float)M_PI * -.05f, 1.0f, "oculus_home.bin"}, // Oculus Home
    {vector3(-0.253644f, 0.577575f, 1.081316f), -0.162111f, -0.453079f, 1.0f, "bunny.bin"},         // Stanford Bunny
    {vector3(10.091616f, 4.139270f, 1.230567f), -5.378105f, -0.398078f, 1.0f, "conference.bin"},    // Conference Room
    {vector3(4.198845f, 6.105420f, -0.400903f), -4.704108f, -0.200078f, .01f, "sponza.bin"},        // Crytek Sponza
    {vector3(2.0f, 2.0f, -0.5f), -(float)M_PI * .5f, (float)M_PI * -.05f, 1.0f, "bistro.bin"}       // Amazon Bistro
};

struct CameraSettings {
    float lensRadius = (ENABLE_DEPTH_OF_FIELD == 1) ? 0.0015f : 0.0f;
    float focalDistance = 0.3f;
    bool foveatedCamera = (ENABLE_FOVEATED == 1);
    bool movable = (DISABLE_MOVEMENT == 0);
    float maxSpeed = (float)CAMERA_SPEED;
};

struct CameraControl {
    vector3 pos = {};
    float yaw = 0.0f;
    float pitch = 0.0f;
    void locallyTranslate(vector3 delta) {
        pos += hvvr::matrix3x3(hvvr::quaternion::fromEulerAngles(yaw, pitch, 0)) * delta;
    }
    hvvr::transform toTransform() const {
        return hvvr::transform(pos, hvvr::quaternion::fromEulerAngles(yaw, pitch, 0), 1.0f);
    }
};

class GApp {
public:
    enum OutputMode { OUTPUT_NONE, OUTPUT_3D_API };
    struct Settings {
        std::string windowName = "Modelviewer";
        uint32_t width = RT_WIDTH;
        uint32_t height = RT_HEIGHT;
        std::string sceneBasePath = "../../../../libraries/hvvr/samples_shared/data/scenes/";
        // 0 = off, 1 = match monitor refresh, 2 = half monitor refresh
        int vSync = ENABLE_VSYNC;
        SceneSpecification initScene;
        OutputMode outputMode = OUTPUT_3D_API;
    };
    GApp(Settings settings);
    virtual void onInit();
    virtual void onShutdown();
    virtual void onUserInput();
    virtual void onSimulation(double sceneTime, double deltaTime);
    virtual void onRender();
    virtual void onLoadScene(SceneSpecification spec);
    virtual void onAfterLoadScene();
    virtual void loadScene(SceneSpecification spec) {
        onLoadScene(spec);
        onAfterLoadScene();
    }
    virtual void endFrame();

    std::unique_ptr<WindowD3D11>& window() {
        return m_window;
    }

    void setResizeCallback(std::function<void(int, int)> callback);

    // Run until we get a quit message, on which we return
    MSG run();

protected:
    Settings m_settings;
    hvvr::Timer m_timer;

    std::unique_ptr<WindowD3D11> m_window;
    std::unique_ptr<hvvr::Raycaster> m_rayCaster;

    std::function<void(int, int)> m_resizeCallback;

    double m_prevElapsedTime;
    double m_deltaTime;

    uint64_t m_frameID = 0;
    hvvr::Camera* m_camera = nullptr;
    CameraControl m_cameraControl = {};

    CameraSettings m_cameraSettings;
};


GApp::GApp(Settings settings) {
    m_settings = settings;
    m_window = std::make_unique<WindowD3D11>(settings.windowName.c_str(), settings.width, settings.height);
    auto resizeCallback = [this](int width, int height) {
        hvvr::ImageViewR8G8B8A8 image((uint32_t*)m_window->getRenderTargetTex(), width, height, width);
        hvvr::ImageResourceDescriptor renderTarget(image);
        renderTarget.memoryType = hvvr::ImageResourceDescriptor::MemoryType::DX_TEXTURE;

        hvvr::DynamicArray<hvvr::Sample> samples = hvvr::getGridSamples(width, height);

        m_camera->setViewport(hvvr::FloatRect{{-(float)width / height, -1}, {(float)width / height, 1}});
        m_camera->setRenderTarget(renderTarget);
        if (ENABLE_WIDE_FOV) {
            m_camera->setSphericalWarpSettings(210.0f, 130.0f);
        }
        m_camera->setSamples(samples.data(), uint32_t(samples.size()), 1);
    };
    setResizeCallback(resizeCallback);

    auto mouseCallback = [this](int dx, int dy) {
        (void)dx;
        (void)dy;
        if (m_cameraSettings.movable) {
            m_cameraControl.yaw += -dx * 0.001f;
            m_cameraControl.pitch += -dy * 0.001f;
        }
    };
    m_window->setRawMouseInputCallback(mouseCallback);


    input::registerDefaultRawInputDevices(m_window->getWindowHandle());
}


void GApp::onSimulation(double sceneTime, double deltaTime) {
    (void)sceneTime;
    hvvr::vector3 posDelta(0.0f);
    if (m_cameraSettings.movable) {
        float cardinalCameraSpeed = (float)(m_cameraSettings.maxSpeed * deltaTime);
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

        m_cameraControl.locallyTranslate(posDelta);
    }

#if GAZE_CURSOR_MODE == GAZE_CURSOR_MODE_MOUSE
    if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) {
        POINT cursorCoord = {};
        GetCursorPos(&cursorCoord);
        ScreenToClient(HWND(m_window->getWindowHandle()), &cursorCoord);

        RECT clientRect = {};
        GetClientRect(HWND(m_window->getWindowHandle()), &clientRect);
        int width = clientRect.right - clientRect.left;
        int height = clientRect.bottom - clientRect.top;

        float cursorX = float(cursorCoord.x) / width * 2.0f - 1.0f;
        float cursorY = -(float(cursorCoord.y) / height * 2.0f - 1.0f);

        if (fabsf(cursorX) <= 1.0f && fabsf(cursorY) <= 1.0f) {
            float screenDistance = 1.0f;
            hvvr::vector3 cursorPosEye(cursorX * width / height, cursorY, -screenDistance);
            hvvr::vector3 eyeDir = hvvr::normalize(cursorPosEye);
            m_camera->setEyeDir(eyeDir);
        }
    }
#endif

}

MSG GApp::run() {
    onInit();
    // The main loop.
    MSG msg;
    for (;;) {
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                goto SHUTDOWN_APP;
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }

        double elapsedTime = m_timer.getElapsed();
        m_deltaTime = elapsedTime - m_prevElapsedTime;
        m_prevElapsedTime = elapsedTime;
        onUserInput();
        onSimulation(m_prevElapsedTime, m_deltaTime);
        onRender();
        endFrame();
    }
SHUTDOWN_APP:
    onShutdown();
    return msg;
}

void GApp::endFrame() {
    if (m_settings.outputMode == OUTPUT_3D_API) {
        uint32_t syncInterval = m_settings.vSync;
        m_window->copyAndPresent(syncInterval);
    }
    // collect some overall perf statistics
    {
        struct FrameStats {
            float deltaTime;
            uint32_t rayCount;
        };
        enum { frameStatsWindowSize = 64 };
        static FrameStats frameStats[frameStatsWindowSize] = {};
        static int frameStatsPos = 0;

        uint32_t rayCount = m_camera->getSampleCount();

        frameStats[frameStatsPos].deltaTime = (float)m_deltaTime;
        frameStats[frameStatsPos].rayCount = rayCount;

        // let it run for a bit before collecting numbers
        if (frameStatsPos == 0 && m_frameID > frameStatsWindowSize * 4) {
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

    m_frameID++;
}

void GApp::setResizeCallback(std::function<void(int, int)> callback) {
    m_resizeCallback = callback;
    m_window->setResizeCallback(callback);
}

void GApp::onAfterLoadScene() {
    // Setup a regular camera
    m_camera = m_rayCaster->createCamera(hvvr::FloatRect(hvvr::vector2(-1, -1), hvvr::vector2(1, 1)),
                                         m_cameraSettings.lensRadius);
    m_camera->setFocalDepth(m_cameraSettings.focalDistance);

    m_resizeCallback(m_window->getWidth(),
                     m_window->getHeight()); // make sure we bind a render target and some samples to the camera
}

void GApp::onLoadScene(SceneSpecification spec) {
    m_cameraControl.pos = spec.defaultCameraOrigin;
    m_cameraControl.yaw = spec.defaultCameraYaw;
    m_cameraControl.pitch = spec.defaultCameraPitch;
    float sceneScale = spec.scale;
    std::string scenePath = m_settings.sceneBasePath + spec.filename;

    // add a default directional light
    hvvr::LightUnion light = {};
    light.type = hvvr::LightType::directional;
    light.directional.Direction = hvvr::normalize(hvvr::vector3(-.25f, 1.0f, 0.1f));
    light.directional.Power = hvvr::vector3(0.4f, 0.35f, 0.35f);
    m_rayCaster->createLight(light);

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
    if (!model_import::createObjects(*m_rayCaster, importedModel)) {
        hvvr::fail("failed to create model objects");
    }
}

void GApp::onInit() {
    RayCasterSpecification spec;
    if (m_cameraSettings.foveatedCamera) {
        spec = RayCasterSpecification::feb2017FoveatedDemoSettings();
    }
    spec.outputTo3DApi = (m_settings.outputMode == OUTPUT_3D_API);
    m_rayCaster = std::make_unique<hvvr::Raycaster>(spec);

    loadScene(m_settings.initScene);
}

void GApp::onShutdown() {
    m_camera = nullptr;
    m_rayCaster = nullptr;
}

void GApp::onUserInput() {}

void GApp::onRender() {
    m_camera->setCameraToWorld(m_cameraControl.toTransform());
    m_rayCaster->render(m_prevElapsedTime);
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

    GApp::Settings settings = {};
    settings.windowName = "HVVR Modelviewer";
    settings.sceneBasePath = "../../../../libraries/hvvr/samples_shared/data/scenes/";

    // The only command line argument is the (optional) scene index
    if (__argc > 1) {
        int sceneIndex = atoi(__argv[1]);
        if (sceneIndex >= 0 && sceneIndex < ModelviewerScene::SceneCount) {
            gSceneSelect = ModelviewerScene(sceneIndex);
            printf("Set Scene index to %d, filename: %s\n", sceneIndex, gSceneSpecs[sceneIndex].filename.c_str());
        }
    }

    if (gSceneSelect < 0 || gSceneSelect >= ModelviewerScene::SceneCount) {
        hvvr::fail("invalid scene enum");
    }

    settings.initScene = gSceneSpecs[gSceneSelect];

    GApp app(settings);
    MSG msg = app.run();
    return (int)msg.wParam;
}
