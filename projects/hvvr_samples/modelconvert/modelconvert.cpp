/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model_import.h"

#include <stdio.h>

#pragma warning(disable: 4702) // unreachable code

int main(int argc, char** argv) {
#if MODEL_IMPORT_ENABLE_FBX == 0
    printf("error: rebuild with MODEL_IMPORT_ENABLE_FBX=1\n");
    return -1;
#endif

    int pathCount = argc - 1;
    if (pathCount <= 0 || pathCount % 2 != 0) {
        printf("usage: modelconvert input_path0 output_path0 [input_path1 output_path1 ...]\n");
        return -1;
    }
    char** paths = argv + 1;
    int modelCount = pathCount / 2;

    for (int n = 0; n < modelCount; n++) {
        const char* inputPath = paths[n * 2 + 0];
        const char* outputPath = paths[n * 2 + 1];

        printf("importing: %s\n", inputPath);
        model_import::Model importedModel;
        if (!model_import::load(inputPath, importedModel)) {
            printf("error: failed to load model %s\n", inputPath);
        }

        printf("saving: %s\n", outputPath);
        if (!model_import::saveBin(outputPath, importedModel)) {
            printf("error: failed to save model %s\n", outputPath);
        }
    }

    return 0;
}
