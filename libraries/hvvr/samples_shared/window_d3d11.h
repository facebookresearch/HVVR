#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <stdint.h>
#include <functional>
struct IDXGISwapChain;
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;
struct ID3D11RenderTargetView;

class WindowD3D11 {
public:
    typedef std::function<void(int,int)>  ResizeCallback;
	typedef std::function<void(int, int)> RawMouseInputCallback;

    WindowD3D11(const char* name,
                uint32_t width,
                uint32_t height,
                ResizeCallback resizeCallback = nullptr,
                RawMouseInputCallback rawMouseInputCallback = nullptr);
    ~WindowD3D11();

	void setResizeCallback(ResizeCallback cb) {
		_resizeCallback = cb;
	}

	void setRawMouseInputCallback(RawMouseInputCallback cb) {
		_rawMouseInputCallback = cb;
	}

    void* getWindowHandle() const {
        return _windowHandle;
    }
    uint32_t getWidth() const {
        return _width;
    }
    uint32_t getHeight() const {
        return _height;
    }

    ID3D11Device* getDevice() const {
        return _device;
    }
    ID3D11DeviceContext* getContext() const {
        return _context;
    }
    IDXGISwapChain* getSwapChain() const {
        return _swapChain;
    }
    ID3D11Texture2D* getBackBufferTex() const {
        return _backBufferTex;
    }
    ID3D11RenderTargetView* getBackBufferRTV() const {
        return _backBufferRTV;
    }
    ID3D11Texture2D* getRenderTargetTex() const {
        return _renderTargetTex;
    }

    // copy the intermediate render target to the backbuffer and present it
    void copyAndPresent(uint32_t syncInterval);
    // just present the backbuffer - don't copy from the intermediate render target
    void present(uint32_t syncInterval);

protected:
    void* _windowHandle;
    uint32_t _width;
    uint32_t _height;
    ResizeCallback _resizeCallback;
    RawMouseInputCallback _rawMouseInputCallback;

    ID3D11Device* _device;
    ID3D11DeviceContext* _context;
    IDXGISwapChain* _swapChain;
    ID3D11Texture2D* _backBufferTex;
    ID3D11RenderTargetView* _backBufferRTV;

    // not the actual backbuffer - we'll render to this and blit to the backbuffer later
    // (CUDA compat)
    ID3D11Texture2D* _renderTargetTex;

    void initRenderTargets();
    void onResize();

public:
    // internal message handler, do not call directly
    void* onMessage(uint32_t uMsg, uintptr_t wParam, intptr_t lParam);
};
