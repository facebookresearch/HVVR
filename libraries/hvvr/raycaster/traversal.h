#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define TRAVERSAL_REF 1
#define TRAVERSAL_AVX 2
#define TRAVERSAL_MODE TRAVERSAL_AVX

#include "dynamic_array.h"
#include "vector_math.h"

#if TRAVERSAL_MODE == TRAVERSAL_REF
# include "traverse_ref.h"
namespace hvvr {
    using namespace traverse::ref;
}
#elif TRAVERSAL_MODE == TRAVERSAL_AVX
# include "traverse_avx.h"
namespace hvvr {
    using namespace traverse::avx;
}
#else
# error unknown traversal mode
#endif

namespace hvvr {

// TODO(anankervis):
// -refactor plane representation such that dot(p, vector4(v, 1)) == 0 (negate the W component)
// -transform() and updatePrecomputed() use mInvTranspose * plane to update plane equations (inc axes of sep)
// --as opposed to recomputing all the cross products and such, use transpose of inverse for normals/planes
inline Frustum frustumTransform(const Frustum& frustum, const matrix4x4& m) {
    vector3 tO[Frustum::pointCount];
    vector3 tD[Frustum::pointCount];
    for (int v = 0; v < Frustum::pointCount; v++) {
        tO[v] = vector3(m * vector4(frustum.pointOrigin[v], 1.0f));
        tD[v] = matrix3x3(m) * frustum.pointDir[v];
    }

    return Frustum(tO, tD);
}

inline bool frustumTestPoint(const Frustum& frustum, const vector3& p) {
    for (int planeIndex = 0; planeIndex < Frustum::planeCount; planeIndex++) {
        if (dot(vector3(frustum.plane[planeIndex]), p) - frustum.plane[planeIndex].w > 0)
            return false;
    }
    return true;
}

struct RayHierarchy {
    ArrayView<const Frustum> blockFrusta;
    ArrayView<const Frustum> tileFrusta;
};

} // namespace hvvr
