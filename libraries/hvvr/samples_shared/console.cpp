/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "console.h"

#include <Windows.h>
#include <stdio.h>


namespace console {

bool createStdOutErr() {
    if (!AllocConsole())
        return false;

    FILE* consoleout;
    freopen_s(&consoleout, "CONOUT$", "wt", stdout);
    freopen_s(&consoleout, "CONOUT$", "wt", stderr);
    SetConsoleTitleA("Debug Console");
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);

    return true;
}

} // namespace console
