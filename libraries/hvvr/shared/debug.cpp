/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "debug.h"

#include <stdarg.h>
#include <stdio.h>

#ifdef _WIN32
# include <Windows.h>
#endif

namespace hvvr {

void fail(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    fprintf(stderr, "%s\n", buffer);
#ifdef _WIN32
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");
    __debugbreak();
#else
    exit(-1);
#endif
}

} // namespace hvvr
