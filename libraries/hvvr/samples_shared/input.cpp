/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "input.h"

#include <Windows.h>


namespace input {

bool registerDefaultRawInputDevices(void* hWnd) {
    RAWINPUTDEVICE rawioDevices[] = {
        {
            /* usUsagePage = */ 0x01,       // top level collection
            /* usUsage     = */ 0x02,       // mouse device
            /* dwFlags     = */ 0x00,       // ignores legacy mouse messages
            /* hwndTarget  = */ HWND(hWnd), // nullptr tracks keyboard focus
        },
        {
            /* usUsagePage = */ 0x01,       // top level collection
            /* usUsage     = */ 0x06,       // keyboard device
            /* dwFlags     = */ 0x00,       // RIDEV_NOLEGACY | RIDEV_APPKEYS
            /* hwndTarget  = */ HWND(hWnd), // nullptr tracks keyboard focus
        },
        {
            /* usUsagePage = */ 0x0c,       // lower level collection
            /* usUsage     = */ 0x01,       // keyboard extensions
            /* dwFlags     = */ 0x00,       // no flags
            /* hwndTarget  = */ HWND(hWnd), // nullptr tracks keyboard focus
        },
    };
    if (!RegisterRawInputDevices(rawioDevices, sizeof(rawioDevices) / sizeof(RAWINPUTDEVICE), sizeof(RAWINPUTDEVICE))) {
        return false;
    }
    return true;
}

} // namespace input
