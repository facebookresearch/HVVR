/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "window_d3d11.h"
#include "debug.h"

#include <assert.h>
#include <d3d11.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")


static __forceinline void validateHR(HRESULT hr) {
    if (FAILED(hr)) {
        hvvr::fail("Failed to validate hr: %08x (%d)", hr, hr);
    }
}

template <typename T>
static void safeRelease(T*& p) {
    if (p == nullptr)
        return;

    p->Release();
    p = nullptr;
}

static LRESULT __stdcall gOnMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    WindowD3D11* window = (WindowD3D11*)GetWindowLongPtr(hWnd, GWLP_USERDATA);
    if (window == nullptr) {
        // we get here during window creation, before we've had a chance to set the user data
        return DefWindowProcA(hWnd, uMsg, wParam, lParam);
    }

    return (LRESULT)window->onMessage(uMsg, wParam, lParam);
}

void* WindowD3D11::onMessage(uint32_t uMsg, uintptr_t wParam, intptr_t lParam) {
    switch (uMsg) {
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE)
                PostQuitMessage(0);
            break;

        case WM_SIZE:
            onResize();
            break;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_INPUT: {
            uint32_t rawSize = 0;
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, nullptr, &rawSize, sizeof(RAWINPUTHEADER));

            char stackBuffer[80];
            char* buffer = rawSize > sizeof(stackBuffer) ? new char [rawSize] : (char*)stackBuffer;

            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, buffer, &rawSize, sizeof(RAWINPUTHEADER));

            auto& input = *(RAWINPUT*)buffer;
            switch (input.header.dwType) {
                case RIM_TYPEKEYBOARD:
                    break;

                case RIM_TYPEMOUSE:
                    if (_rawMouseInputCallback) {
                        _rawMouseInputCallback(input.data.mouse.lLastX, input.data.mouse.lLastY);
                    }
                    break;
            }

            if (buffer != stackBuffer)
                delete [] buffer;
        } break;
    }

    return (void*)DefWindowProcA(HWND(_windowHandle), uMsg, wParam, lParam);
}

WindowD3D11::WindowD3D11(const char* name,
                         uint32_t width,
                         uint32_t height,
                         ResizeCallback resizeCallback,
                         RawMouseInputCallback rawMouseInputCallback)
    : _windowHandle(nullptr)
    , _width(width)
    , _height(height)
    , _resizeCallback(resizeCallback)
    , _rawMouseInputCallback(rawMouseInputCallback)
    , _device(nullptr)
    , _context(nullptr)
    , _swapChain(nullptr)
    , _backBufferTex(nullptr)
    , _backBufferRTV(nullptr)
    , _renderTargetTex(nullptr) {
    const char* className = "WindowD3D11";

    WNDCLASSEXA wcex = {};
    wcex.cbSize = sizeof(WNDCLASSEXA);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = gOnMessage;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = GetModuleHandleA(nullptr);
    wcex.hIcon = 0;
    wcex.hCursor = LoadCursorA(nullptr, IDC_ARROW);
    wcex.hbrBackground = nullptr;
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = className;
    wcex.hIconSm = 0;
    if (!RegisterClassExA(&wcex)) {
        hvvr::fail("RegisterClassExA failed with %x\n", GetLastError());
    }

    RECT windowRect = {};
    windowRect.right = _width;
    windowRect.bottom = _height;
    DWORD windowStyle = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&windowRect, windowStyle, false);
    _windowHandle = (void*)CreateWindowExA(0, className, name, windowStyle, CW_USEDEFAULT, CW_USEDEFAULT,
                                           windowRect.right - windowRect.left, windowRect.bottom - windowRect.top,
                                           nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
    if (_windowHandle == nullptr) {
        hvvr::fail("CreateWindowExA failed with %x\n", GetLastError());
    }
    SetWindowLongPtr(HWND(_windowHandle), GWLP_USERDATA, LONG_PTR(this));
    ShowWindow(HWND(_windowHandle), SW_SHOW);
    UpdateWindow(HWND(_windowHandle));

    // check that we actually got the client rect we asked for
    RECT actualClientRect;
    GetClientRect(HWND(_windowHandle), &actualClientRect);
    if (actualClientRect.right != int(_width) || actualClientRect.bottom != int(_height)) {
        // for some reason, passing the output of AdjustWindowRect into SetWindowPos still produces a client rect a bit
        // too small (but it does work for CreateWindowEx)
        RECT actualWindowRect;
        GetWindowRect(HWND(_windowHandle), &actualWindowRect);
        int adjustedWidth = actualWindowRect.right - actualWindowRect.left + (_width - actualClientRect.right);
        int adjustedHeight = actualWindowRect.bottom - actualWindowRect.top + (_height - actualClientRect.bottom);

        // SWP_NOSENDCHANGING is the magic to avoid Windows clamping the window size to the desktop size
        SetWindowPos(
            HWND(_windowHandle), HWND_TOPMOST, 0, 0, adjustedWidth, adjustedHeight,
            SWP_NOACTIVATE | SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_NOSENDCHANGING);

        GetClientRect(HWND(_windowHandle), &actualClientRect);
        if (actualClientRect.right != int(_width) || actualClientRect.bottom != int(_height)) {
            assert(false);
        }
    }

    IDXGIFactory1* factory = nullptr;
    validateHR(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)(&factory)));

    IDXGIAdapter1* adapter = nullptr;
    validateHR(factory->EnumAdapters1(0, &adapter));

    uint32_t d3d11CreateFlags = 0;
    d3d11CreateFlags |= D3D11_CREATE_DEVICE_SINGLETHREADED; // we will only call d3d11 from a single thread
    d3d11CreateFlags |= D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS; // no extra driver threads
    validateHR(D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, d3d11CreateFlags, nullptr, 0,
                                 D3D11_SDK_VERSION, &_device, nullptr, &_context));
    safeRelease(adapter);

    RECT clientRect = {};
    GetClientRect(HWND(_windowHandle), &clientRect);

    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferDesc.Width = uint32_t(clientRect.right);
    swapChainDesc.BufferDesc.Height = uint32_t(clientRect.bottom);
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 0;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER;
    swapChainDesc.BufferCount = 2;
    swapChainDesc.OutputWindow = HWND(_windowHandle);
    swapChainDesc.Windowed = TRUE;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    validateHR(factory->CreateSwapChain(_device, &swapChainDesc, &_swapChain));
    safeRelease(factory);

    initRenderTargets();
}

WindowD3D11::~WindowD3D11() {
    safeRelease(_renderTargetTex);
    safeRelease(_backBufferRTV);
    safeRelease(_backBufferTex);
    safeRelease(_swapChain);
    safeRelease(_context);
    safeRelease(_device);
}

void WindowD3D11::onResize() {
    if (_swapChain == nullptr) {
        // we get here during window creation, prior to creating the d3d resources
        return;
    }

    // must release references to the swap chain before resizing it
    safeRelease(_backBufferTex);
    safeRelease(_backBufferRTV);
    validateHR(_swapChain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH));

    initRenderTargets();

    if (_resizeCallback) {
        _resizeCallback(_width, _height);
    }
}

void WindowD3D11::initRenderTargets() {
    safeRelease(_backBufferTex);
    validateHR(_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&_backBufferTex));

    D3D11_TEXTURE2D_DESC bbDesc = {};
    _backBufferTex->GetDesc(&bbDesc);
    _width = bbDesc.Width;
    _height = bbDesc.Height;

    safeRelease(_backBufferRTV);
    validateHR(_device->CreateRenderTargetView(_backBufferTex, nullptr, &_backBufferRTV));

    safeRelease(_renderTargetTex);
    {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = _width;
        desc.Height = _height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;
        validateHR(_device->CreateTexture2D(&desc, nullptr, &_renderTargetTex));
    }
}

void WindowD3D11::copyAndPresent(uint32_t syncInterval) {
    _context->CopyResource(_backBufferTex, _renderTargetTex);
    present(syncInterval);
}

void WindowD3D11::present(uint32_t syncInterval) {
    _swapChain->Present(syncInterval, 0);
}
