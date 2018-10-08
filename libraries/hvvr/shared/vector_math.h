#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "cuda_decl.h"
#include "util.h"

// CUDA vector types
#include <vector_types.h>
#if CUDA_COMPILE
#include <cuda_fp16.h>
#include <vector_functions.h>
#endif

#include <math.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h> // _CountLeadingZeros
#endif

namespace hvvr {

struct vector2 {
    float x, y;

    CHD vector2() {}
    CHD vector2(float _x, float _y) : x(_x), y(_y) {}
    CHD explicit vector2(float _x) : x(_x), y(_x) {}
    CHD explicit vector2(const float* v) : x(v[0]), y(v[1]) {}

    // compat with CUDA built-in types
    CHD explicit vector2(const float2& v) : x(v.x), y(v.y) {}
    CHD explicit operator float2() const {
        return {x, y};
    }

    CHD vector2 operator-() const {
        return vector2(-x, -y);
    }

    CHD vector2 operator+(const vector2& v) const {
        return vector2(x + v.x, y + v.y);
    }
    CHD vector2 operator-(const vector2& v) const {
        return vector2(x - v.x, y - v.y);
    }
    CHD vector2 operator*(const vector2& v) const {
        return vector2(x * v.x, y * v.y);
    }
    CHD vector2 operator/(const vector2& v) const {
        return vector2(x / v.x, y / v.y);
    }

    CHD vector2 operator+(float a) const {
        return vector2(x + a, y + a);
    }
    CHD vector2 operator-(float a) const {
        return vector2(x - a, y - a);
    }
    CHD vector2 operator*(float a) const {
        return vector2(x * a, y * a);
    }
    CHD vector2 operator/(float a) const {
        float aInv = 1.0f / a;
        return vector2(x * aInv, y * aInv);
    }

    CHD vector2& operator+=(const vector2& v) {
        return *this = *this + v;
    }
    CHD vector2& operator-=(const vector2& v) {
        return *this = *this - v;
    }
    CHD vector2& operator*=(const vector2& v) {
        return *this = *this * v;
    }
    CHD vector2& operator/=(const vector2& v) {
        return *this = *this / v;
    }

    CHD vector2& operator+=(float a) {
        return *this = *this + a;
    }
    CHD vector2& operator-=(float a) {
        return *this = *this - a;
    }
    CHD vector2& operator*=(float a) {
        return *this = *this * a;
    }
    CHD vector2& operator/=(float a) {
        return *this = *this / a;
    }

    CHD bool operator==(const vector2& v) const {
        return x == v.x && y == v.y;
    }
    CHD bool operator!=(const vector2& v) const {
        return x != v.x || y != v.y;
    }
};

CHDI vector2 operator+(float a, const vector2& b) {
    return b + a;
}
CHDI vector2 operator*(float a, const vector2& b) {
    return b * a;
}

CHDI float dot(const vector2& a, const vector2& b) {
    return a.x * b.x + a.y * b.y;
}
CHDI float lengthSq(const vector2& v) {
    return dot(v, v);
}
CHDI float length(const vector2& v) {
    return sqrtf(lengthSq(v));
}
CHDI vector2 normalize(const vector2& v) {
    return v / length(v);
}
CHDI vector2 normalizeSafe(const vector2& v) {
    float mag = length(v);
    if (mag > 0.0f) {
        return v / mag;
    }
    return v;
}
CHDI float cross(const vector2& a, const vector2& b) {
    return a.x * b.y - a.y * b.x;
}

CHDI vector2 min(const vector2& a, const vector2& b) {
    return vector2(min(a.x, b.x), min(a.y, b.y));
}
CHDI vector2 min(const vector2& a, float b) {
    return vector2(min(a.x, b), min(a.y, b));
}
CHDI vector2 max(const vector2& a, const vector2& b) {
    return vector2(max(a.x, b.x), max(a.y, b.y));
}
CHDI vector2 max(const vector2& a, float b) {
    return vector2(max(a.x, b), max(a.y, b));
}
CHDI vector2 clamp(const vector2& a, const vector2& lower, const vector2& upper) {
    return vector2(clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y));
}
CHDI vector2 clamp(const vector2& a, float lower, float upper) {
    return vector2(clamp(a.x, lower, upper), clamp(a.y, lower, upper));
}
CHDI vector2 abs(const vector2& v) {
    return vector2(fabsf(v.x), fabsf(v.y));
}

// TODO(anankervis): create wrapper types for dir3 vs pos3/point3 to enable implicit
// conversion to vector4 with w=0 or w=1?
// -see also:
// --transform::operator*(const vector3&) (should have different behavior for direction vs position)
struct vector3 {
    float x, y, z;

    CHD vector3() {}
    CHD vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    CHD explicit vector3(float _x) : x(_x), y(_x), z(_x) {}
    CHD explicit vector3(const vector2& v, float _z = 0.0f) : x(v.x), y(v.y), z(_z) {}
    CHD explicit vector3(const float* v) : x(v[0]), y(v[1]), z(v[2]) {}

    CHD explicit operator vector2() const {
        return vector2(x, y);
    }

    // compat with CUDA built-in types
    CHD explicit vector3(const float3& v) : x(v.x), y(v.y), z(v.z) {}
    CHD explicit operator float3() const {
        return {x, y, z};
    }

    CHD vector3 operator-() const {
        return vector3(-x, -y, -z);
    }

    CHD vector3 operator+(const vector3& v) const {
        return vector3(x + v.x, y + v.y, z + v.z);
    }
    CHD vector3 operator-(const vector3& v) const {
        return vector3(x - v.x, y - v.y, z - v.z);
    }
    CHD vector3 operator*(const vector3& v) const {
        return vector3(x * v.x, y * v.y, z * v.z);
    }
    CHD vector3 operator/(const vector3& v) const {
        return vector3(x / v.x, y / v.y, z / v.z);
    }

    CHD vector3 operator+(float a) const {
        return vector3(x + a, y + a, z + a);
    }
    CHD vector3 operator-(float a) const {
        return vector3(x - a, y - a, z - a);
    }
    CHD vector3 operator*(float a) const {
        return vector3(x * a, y * a, z * a);
    }
    CHD vector3 operator/(float a) const {
        float aInv = 1.0f / a;
        return vector3(x * aInv, y * aInv, z * aInv);
    }

    CHD vector3& operator+=(const vector3& v) {
        return *this = *this + v;
    }
    CHD vector3& operator-=(const vector3& v) {
        return *this = *this - v;
    }
    CHD vector3& operator*=(const vector3& v) {
        return *this = *this * v;
    }
    CHD vector3& operator/=(const vector3& v) {
        return *this = *this / v;
    }

    CHD vector3& operator+=(float a) {
        return *this = *this + a;
    }
    CHD vector3& operator-=(float a) {
        return *this = *this - a;
    }
    CHD vector3& operator*=(float a) {
        return *this = *this * a;
    }
    CHD vector3& operator/=(float a) {
        return *this = *this / a;
    }

    CHD bool operator==(const vector3& v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    CHD bool operator!=(const vector3& v) const {
        return x != v.x || y != v.y || z != v.z;
    }

    // Don't use in kernels!
    CUDA_HOST const float& operator[](size_t index) const {
        return *(&x + index);
    }
    CUDA_HOST float& operator[](size_t index) {
        return *(&x + index);
    }
};

CHDI vector3 operator+(float a, const vector3& b) {
    return b + a;
}
CHDI vector3 operator*(float a, const vector3& b) {
    return b * a;
}

CHDI float dot(const vector3& a, const vector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
CHDI float lengthSq(const vector3& v) {
    return dot(v, v);
}
CHDI float length(const vector3& v) {
    return sqrtf(lengthSq(v));
}
CHDI vector3 normalize(const vector3& v) {
    return v / length(v);
}
CHDI vector3 normalizeSafe(const vector3& v) {
    float mag = length(v);
    if (mag > 0.0f) {
        return v / mag;
    }
    return v;
}
CHDI vector3 cross(const vector3& a, const vector3& b) {
    return vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

CHDI vector3 min(const vector3& a, const vector3& b) {
    return vector3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
CHDI vector3 min(const vector3& a, float b) {
    return vector3(min(a.x, b), min(a.y, b), min(a.z, b));
}
CHDI vector3 max(const vector3& a, const vector3& b) {
    return vector3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
CHDI vector3 max(const vector3& a, float b) {
    return vector3(max(a.x, b), max(a.y, b), max(a.z, b));
}
CHDI vector3 clamp(const vector3& a, const vector3& lower, const vector3& upper) {
    return vector3(clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y), clamp(a.z, lower.z, upper.z));
}
CHDI vector3 clamp(const vector3& a, float lower, float upper) {
    return vector3(clamp(a.x, lower, upper), clamp(a.y, lower, upper), clamp(a.z, lower, upper));
}
CHDI vector3 abs(const vector3& v) {
    return vector3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

struct vector4 {
    float x, y, z, w;

    CHD vector4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {} // TODO(anankervis): legacy - zero initialize, remove this
    CHD vector4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
    CHD vector4(const vector2& a, const vector2& b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
    CHD explicit vector4(float _x) : x(_x), y(_x), z(_x), w(_x) {}
    CHD explicit vector4(const vector2& v, float _z = 0.0f, float _w = 0.0f) : x(v.x), y(v.y), z(_z), w(_w) {}
    CHD explicit vector4(const vector3& v, float _w = 0.0f) : x(v.x), y(v.y), z(v.z), w(_w) {}
    CHD explicit vector4(const float* v) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {}

    CHD explicit operator vector2() const {
        return vector2(x, y);
    }
    CHD explicit operator vector3() const {
        return vector3(x, y, z);
    }

    // compat with CUDA built-in types
    CHD explicit vector4(const float4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
    CHD explicit operator float4() const {
        return {x, y, z, w};
    }

    CHD vector4 operator-() const {
        return vector4(-x, -y, -z, -w);
    }

    CHD vector4 operator+(const vector4& v) const {
        return vector4(x + v.x, y + v.y, z + v.z, w + v.w);
    }
    CHD vector4 operator-(const vector4& v) const {
        return vector4(x - v.x, y - v.y, z - v.z, w - v.w);
    }
    CHD vector4 operator*(const vector4& v) const {
        return vector4(x * v.x, y * v.y, z * v.z, w * v.w);
    }
    CHD vector4 operator/(const vector4& v) const {
        return vector4(x / v.x, y / v.y, z / v.z, w / v.w);
    }

    CHD vector4 operator+(float a) const {
        return vector4(x + a, y + a, z + a, w + a);
    }
    CHD vector4 operator-(float a) const {
        return vector4(x - a, y - a, z - a, w - a);
    }
    CHD vector4 operator*(float a) const {
        return vector4(x * a, y * a, z * a, w * a);
    }
    CHD vector4 operator/(float a) const {
        float aInv = 1.0f / a;
        return vector4(x * aInv, y * aInv, z * aInv, w * aInv);
    }

    CHD vector4& operator+=(const vector4& v) {
        return *this = *this + v;
    }
    CHD vector4& operator-=(const vector4& v) {
        return *this = *this - v;
    }
    CHD vector4& operator*=(const vector4& v) {
        return *this = *this * v;
    }
    CHD vector4& operator/=(const vector4& v) {
        return *this = *this / v;
    }

    CHD vector4& operator+=(float a) {
        return *this = *this + a;
    }
    CHD vector4& operator-=(float a) {
        return *this = *this - a;
    }
    CHD vector4& operator*=(float a) {
        return *this = *this * a;
    }
    CHD vector4& operator/=(float a) {
        return *this = *this / a;
    }

    CHD bool operator==(const vector4& v) const {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }
    CHD bool operator!=(const vector4& v) const {
        return x != v.x || y != v.y || z != v.z || w != v.w;
    }
};

CHDI vector4 operator+(float a, const vector4& b) {
    return b + a;
}
CHDI vector4 operator*(float a, const vector4& b) {
    return b * a;
}

CHDI float dot(const vector4& a, const vector4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
CHDI float lengthSq(const vector4& v) {
    return dot(v, v);
}
CHDI float length(const vector4& v) {
    return sqrtf(lengthSq(v));
}
CHDI vector4 normalize(const vector4& v) {
    return v / length(v);
}
CHDI vector4 normalizeSafe(const vector4& v) {
    float mag = length(v);
    if (mag > 0.0f) {
        return v / mag;
    }
    return v;
}

CHDI vector4 min(const vector4& a, const vector4& b) {
    return vector4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}
CHDI vector4 min(const vector4& a, float b) {
    return vector4(min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b));
}
CHDI vector4 max(const vector4& a, const vector4& b) {
    return vector4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}
CHDI vector4 max(const vector4& a, float b) {
    return vector4(max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b));
}
CHDI vector4 clamp(const vector4& a, const vector4& lower, const vector4& upper) {
    return vector4(clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y), clamp(a.z, lower.z, upper.z),
                   clamp(a.w, lower.w, upper.w));
}
CHDI vector4 clamp(const vector4& a, float lower, float upper) {
    return vector4(clamp(a.x, lower, upper), clamp(a.y, lower, upper), clamp(a.z, lower, upper),
                   clamp(a.w, lower, upper));
}
CHDI vector4 abs(const vector4& v) {
    return vector4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

struct half {
    uint16_t v;

    CHD half() {}
    CHD explicit half(float x) : v(floatToHalf(x)) {}
    CHD explicit half(uint16_t x) : v(x) {}

    CHD explicit operator float() const {
        return halfToFloat(v);
    }
    CHD explicit operator uint16_t() const {
        return v;
    }

    CHD half operator-() const {
        return half(uint16_t(v ^ 0x8000));
    }

private:
    CHD static uint16_t floatToHalf(float x) {
#if CUDA_COMPILE
        return __half_raw(__float2half(x)).x;
#else
#ifdef _MSC_VER
        return _mm_cvtps_ph(_mm_set_ps1(x), 0).m128i_u16[0];
#else
        return _cvtss_sh(x, 0);
#endif
#endif
    }

    CHD static float halfToFloat(uint16_t x) {
#if CUDA_COMPILE
        return __half2float({x});
#else
#ifdef _MSC_VER
        return _mm_cvtph_ps(_mm_set1_epi16(x)).m128_f32[0];
#else
        return _cvtsh_ss(x);
#endif
#endif
    }
};

struct vector2h {
    half x, y;

    CHD vector2h() {}
    CHD vector2h(half _x, half _y) : x(_x), y(_y) {}
    // TODO(anankervis): use packed half conversion intrinsics
    CHD explicit vector2h(const vector2& v) : x(v.x), y(v.y) {}

    CHD explicit operator vector2() const {
        return vector2(float(x), float(y));
    }

    CHD vector2h operator-() const {
        return vector2h(-x, -y);
    }
};

struct vector4h {
    half x, y, z, w;

    CHD vector4h() {}
    CHD vector4h(half _x, half _y, half _z, half _w) : x(_x), y(_y), z(_z), w(_w) {}
    CHD vector4h(const vector2h& a, const vector2h& b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
    // TODO(anankervis): use packed half conversion intrinsics
    CHD explicit vector4h(const vector3& v) : x(v.x), y(v.y), z(v.z), w(0.0f) {}
    CHD explicit vector4h(const vector4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    CHD explicit operator vector3() const {
        return vector3(float(x), float(y), float(z));
    }
    CHD explicit operator vector4() const {
        return vector4(float(x), float(y), float(z), float(w));
    }

    CHD vector4h operator-() const {
        return vector4h(-x, -y, -z, -w);
    }
};

struct vector2i {
    int32_t x, y;

    CHD vector2i() {}
    CHD vector2i(int32_t _x, int32_t _y) : x(_x), y(_y) {}
    CHD explicit vector2i(int32_t _x) : x(_x), y(_x) {}
    CHD explicit vector2i(const int32_t* v) : x(v[0]), y(v[1]) {}
    CHD explicit vector2i(const vector2& v) : x(int32_t(v.x)), y(int32_t(v.y)) {}

    CHD explicit operator vector2() const {
        return vector2(float(x), float(y));
    }

    // compat with CUDA built-in types
    CHD explicit vector2i(const int2& v) : x(v.x), y(v.y) {}
    CHD explicit operator int2() const {
        return {x, y};
    }

    CHD vector2i operator-() const {
        return vector2i(-x, -y);
    }

    CHD vector2i operator+(const vector2i& v) const {
        return vector2i(x + v.x, y + v.y);
    }
    CHD vector2i operator-(const vector2i& v) const {
        return vector2i(x - v.x, y - v.y);
    }
    CHD vector2i operator*(const vector2i& v) const {
        return vector2i(x * v.x, y * v.y);
    }
    CHD vector2i operator/(const vector2i& v) const {
        return vector2i(x / v.x, y / v.y);
    }

    CHD vector2i operator+(int32_t a) const {
        return vector2i(x + a, y + a);
    }
    CHD vector2i operator-(int32_t a) const {
        return vector2i(x - a, y - a);
    }
    CHD vector2i operator*(int32_t a) const {
        return vector2i(x * a, y * a);
    }
    CHD vector2i operator/(int32_t a) const {
        return vector2i(x / a, y / a);
    }

    CHD vector2i& operator+=(const vector2i& v) {
        return *this = *this + v;
    }
    CHD vector2i& operator-=(const vector2i& v) {
        return *this = *this - v;
    }
    CHD vector2i& operator*=(const vector2i& v) {
        return *this = *this * v;
    }
    CHD vector2i& operator/=(const vector2i& v) {
        return *this = *this / v;
    }

    CHD vector2i& operator+=(int32_t a) {
        return *this = *this + a;
    }
    CHD vector2i& operator-=(int32_t a) {
        return *this = *this - a;
    }
    CHD vector2i& operator*=(int32_t a) {
        return *this = *this * a;
    }
    CHD vector2i& operator/=(int32_t a) {
        return *this = *this / a;
    }

    CHD bool operator==(const vector2i& v) const {
        return x == v.x && y == v.y;
    }
    CHD bool operator!=(const vector2i& v) const {
        return x != v.x || y != v.y;
    }
};

CHDI vector2i operator+(int32_t a, const vector2i& b) {
    return b + a;
}
CHDI vector2i operator*(int32_t a, const vector2i& b) {
    return b * a;
}

CHDI int32_t dot(const vector2i& a, const vector2i& b) {
    return a.x * b.x + a.y * b.y;
}
CHDI int32_t lengthSq(const vector2i& v) {
    return dot(v, v);
}
CHDI int32_t cross(const vector2i& a, const vector2i& b) {
    return a.x * b.y - a.y * b.x;
}

CHDI vector2i min(const vector2i& a, const vector2i& b) {
    return vector2i(min(a.x, b.x), min(a.y, b.y));
}
CHDI vector2i min(const vector2i& a, int32_t b) {
    return vector2i(min(a.x, b), min(a.y, b));
}
CHDI vector2i max(const vector2i& a, const vector2i& b) {
    return vector2i(max(a.x, b.x), max(a.y, b.y));
}
CHDI vector2i max(const vector2i& a, int32_t b) {
    return vector2i(max(a.x, b), max(a.y, b));
}
CHDI vector2i clamp(const vector2i& a, const vector2i& lower, const vector2i& upper) {
    return vector2i(clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y));
}
CHDI vector2i clamp(const vector2i& a, int32_t lower, int32_t upper) {
    return vector2i(clamp(a.x, lower, upper), clamp(a.y, lower, upper));
}
CHDI vector2i abs(const vector2i& v) {
    return vector2i(::abs(v.x), ::abs(v.y));
}

struct vector2ui {
    uint32_t x, y;

    CHD vector2ui() {}
    CHD vector2ui(uint32_t _x, uint32_t _y) : x(_x), y(_y) {}
    CHD explicit vector2ui(uint32_t _x) : x(_x), y(_x) {}
    CHD explicit vector2ui(const uint32_t* v) : x(v[0]), y(v[1]) {}
    CHD explicit vector2ui(const vector2& v) : x(uint32_t(v.x)), y(uint32_t(v.y)) {}
    CHD explicit vector2ui(const vector2i& v) : x(uint32_t(v.x)), y(uint32_t(v.y)) {}

    CHD explicit operator vector2() const {
        return vector2(float(x), float(y));
    }
    CHD explicit operator vector2i() const {
        return vector2i(int32_t(x), int32_t(y));
    }

    // compat with CUDA built-in types
    CHD explicit vector2ui(const uint2& v) : x(v.x), y(v.y) {}
    CHD explicit operator uint2() const {
        return {x, y};
    }

    CHD vector2ui operator+(const vector2ui& v) const {
        return vector2ui(x + v.x, y + v.y);
    }
    CHD vector2ui operator-(const vector2ui& v) const {
        return vector2ui(x - v.x, y - v.y);
    }
    CHD vector2ui operator*(const vector2ui& v) const {
        return vector2ui(x * v.x, y * v.y);
    }
    CHD vector2ui operator/(const vector2ui& v) const {
        return vector2ui(x / v.x, y / v.y);
    }

    CHD vector2ui operator+(uint32_t a) const {
        return vector2ui(x + a, y + a);
    }
    CHD vector2ui operator-(uint32_t a) const {
        return vector2ui(x - a, y - a);
    }
    CHD vector2ui operator*(uint32_t a) const {
        return vector2ui(x * a, y * a);
    }
    CHD vector2ui operator/(uint32_t a) const {
        return vector2ui(x / a, y / a);
    }

    CHD vector2ui& operator+=(const vector2ui& v) {
        return *this = *this + v;
    }
    CHD vector2ui& operator-=(const vector2ui& v) {
        return *this = *this - v;
    }
    CHD vector2ui& operator*=(const vector2ui& v) {
        return *this = *this * v;
    }
    CHD vector2ui& operator/=(const vector2ui& v) {
        return *this = *this / v;
    }

    CHD vector2ui& operator+=(uint32_t a) {
        return *this = *this + a;
    }
    CHD vector2ui& operator-=(uint32_t a) {
        return *this = *this - a;
    }
    CHD vector2ui& operator*=(uint32_t a) {
        return *this = *this * a;
    }
    CHD vector2ui& operator/=(uint32_t a) {
        return *this = *this / a;
    }

    CHD bool operator==(const vector2ui& v) const {
        return x == v.x && y == v.y;
    }
    CHD bool operator!=(const vector2ui& v) const {
        return x != v.x || y != v.y;
    }
};

CHDI vector2ui operator+(uint32_t a, const vector2ui& b) {
    return b + a;
}
CHDI vector2ui operator*(uint32_t a, const vector2ui& b) {
    return b * a;
}

CHDI uint32_t dot(const vector2ui& a, const vector2ui& b) {
    return a.x * b.x + a.y * b.y;
}
CHDI uint32_t lengthSq(const vector2ui& v) {
    return dot(v, v);
}

CHDI vector2ui min(const vector2ui& a, const vector2ui& b) {
    return vector2ui(min(a.x, b.x), min(a.y, b.y));
}
CHDI vector2ui min(const vector2ui& a, uint32_t b) {
    return vector2ui(min(a.x, b), min(a.y, b));
}
CHDI vector2ui max(const vector2ui& a, const vector2ui& b) {
    return vector2ui(max(a.x, b.x), max(a.y, b.y));
}
CHDI vector2ui max(const vector2ui& a, uint32_t b) {
    return vector2ui(max(a.x, b), max(a.y, b));
}
CHDI vector2ui clamp(const vector2ui& a, const vector2ui& lower, const vector2ui& upper) {
    return vector2ui(clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y));
}
CHDI vector2ui clamp(const vector2ui& a, uint32_t lower, uint32_t upper) {
    return vector2ui(clamp(a.x, lower, upper), clamp(a.y, lower, upper));
}

struct quaternion {
    vector4 v;

    CHD quaternion() : v(identity().v) {} // TODO(anankervis): legacy - identity init
    CHD quaternion(float x, float y, float z, float w) : v(x, y, z, w) {}
    CHD explicit quaternion(const vector4& q) : v(q) {}
    CHD explicit quaternion(const float* q) : v(q) {}

    CHD explicit operator vector4() const {
        return v;
    }

    CHD static quaternion identity() { // identity (zero rotation) and multiplicative identity
        return quaternion(0.0f, 0.0f, 0.0f, 1.0f);
    }

    CHD static quaternion rotateX(float radians) {
        return quaternion(sinf(radians * 0.5f), 0, 0, cosf(radians * 0.5f));
    }
    CHD static quaternion rotateY(float radians) {
        return quaternion(0, sinf(radians * 0.5f), 0, cosf(radians * 0.5f));
    }
    CHD static quaternion rotateZ(float radians) {
        return quaternion(0, 0, sinf(radians * 0.5f), cosf(radians * 0.5f));
    }

    // Construct a quaternion from euler angles, assuming -Z is forward and Y is up.
    CHD static quaternion fromEulerAngles(float yawRadians, float pitchRadians, float rollRadians) {
        float cya = cosf(yawRadians * 0.5f);
        float cpi = cosf(pitchRadians * 0.5f);
        float cro = cosf(rollRadians * 0.5f);
        float sya = sinf(yawRadians * 0.5f);
        float spi = sinf(pitchRadians * 0.5f);
        float sro = sinf(rollRadians * 0.5f);
        return quaternion(cro * cya * spi + sro * sya * cpi, cro * sya * cpi - sro * cya * spi,
                          sro * cya * cpi - cro * sya * spi, cro * cya * cpi + sro * sya * spi);
    }

    // Construct a quaternion from an axis and angle (in radians). The axis must be a unit vector.
    CHD static quaternion fromAxisAngle(const vector3& axis, float radians) {
        float c = cosf(radians * 0.5f);
        float s = sinf(radians * 0.5f);
        return quaternion(vector4(axis * s, c));
    }

    // Given a rotation vector of form unitRotationAxis * angle (in radians)
    CHD static quaternion fromRotationVector(const vector3& v) {
        float angleSquared = lengthSq(v);
        float s = 0.0f;
        float c = 1.0f;
        if (angleSquared > 0.0f) {
            float angle = sqrtf(angleSquared);
            s = sinf(angle * 0.5f) / angle; // normalize
            c = cosf(angle * 0.5f);
        }
        return quaternion(vector4(v * s, c));
    }

    CHD static quaternion fromForwardAndUp(vector3 forward, vector3 up) {
        forward = normalize(forward);
        up = normalize(up);
        vector3 right = normalize(cross(up, forward));
        up = normalize(cross(forward, right));
        float m00 = right.x;
        float m01 = right.y;
        float m02 = right.z;
        float m10 = up.x;
        float m11 = up.y;
        float m12 = up.z;
        float m20 = forward.x;
        float m21 = forward.y;
        float m22 = forward.z;
        float n = (m00 + m11) + m22;
        if (n > 0.0f) {
            float a = sqrtf(n + 1.0f);
            float b = 0.5f / a;
            return quaternion((m12 - m21) * b, (m20 - m02) * b, (m01 - m10) * b, a * 0.5f);
        }
        if ((m00 >= m11) && (m00 >= m22)) {
            float a = sqrtf(((1.0f + m00) - m11) - m22);
            float b = 0.5f / a;
            return quaternion(0.5f * a, (m01 + m10) * b, (m02 + m20) * b, (m12 - m21) * b);
        }
        if (m11 > m22) {
            float a = sqrtf(((1.0f + m11) - m00) - m22);
            float b = 0.5f / a;
            return quaternion((m10 + m01) * b, 0.5f * a, (m21 + m12) * b, (m20 - m02) * b);
        } else {
            float a = sqrtf(((1.0f + m22) - m00) - m11);
            float b = 0.5f / a;
            return quaternion((m20 + m02) * b, (m21 + m12) * b, 0.5f * a, (m01 - m10) * b);
        }
    }

    // assumes the quaternion is normalized
    CHD vector3 toEulerAngles() const {
        vector3 euler;
        float ySq = v.y * v.y;

        // roll (x-axis rotation)
        float t0 = 2.0f * (v.w * v.x + v.y * v.z);
        float t1 = 1.0f - 2.0f * (v.x * v.x + ySq);
        euler.x = atan2f(t0, t1);

        // pitch (y-axis rotation)
        float t2 = 2.0f * (v.w * v.y - v.z * v.x);
        if (t2 > 1.0f)
            t2 = 1.0f;
        if (t2 < -1.0f)
            t2 = -1.0f;
        euler.y = asinf(t2);

        // yaw (z-axis rotation)
        float t3 = 2.0f * (v.w * v.z + v.x * v.y);
        float t4 = 1.0f - 2.0f * (ySq + v.z * v.z);
        euler.z = atan2f(t3, t4);

        return euler;
    }

    // assumes the quaternion is normalized
    CHD vector4 toAxisAngle() const {
        float mag = sqrtf(1.0f - v.w * v.w);
        if (mag < 0.000001f) {
            return vector4(1.0f, 0.0f, 0.0f, 0.0f);
        }
        return vector4(vector3(v) / mag, 2.0f * acosf(v.w));
    }

    // negate - produces the same rotation, with different handedness (long vs short way around)
    CHD quaternion operator-() const {
        return quaternion(-v);
    }

    // conjugate (inverse, if the quaternion is normalized)
    CHD quaternion operator~() const {
        return quaternion(-v.x, -v.y, -v.z, v.w);
    }

    CHD quaternion operator*(const quaternion& q) const {
        return quaternion(v.w * q.v.x + v.x * q.v.w + v.y * q.v.z - v.z * q.v.y,
                          v.w * q.v.y - v.x * q.v.z + v.y * q.v.w + v.z * q.v.x,
                          v.w * q.v.z + v.x * q.v.y - v.y * q.v.x + v.z * q.v.w,
                          v.w * q.v.w - (v.x * q.v.x + v.y * q.v.y + v.z * q.v.z));
    }
    CHD quaternion& operator*=(const quaternion& q) {
        return *this = *this * q;
    }

    // assumes normalized quaternion
    CHD vector3 operator*(const vector3& dir) const {
        return vector3(vector4(*this * quaternion(vector4(dir)) * ~*this));
    }

    CHD bool operator==(const quaternion& q) const {
        return v == q.v;
    }
    CHD bool operator!=(const quaternion& q) const {
        return v != q.v;
    }
};

CHDI float lengthSq(const quaternion& q) {
    return lengthSq(q.v);
}
CHDI float length(const quaternion& q) {
    return length(q.v);
}
CHDI quaternion normalize(const quaternion& q) {
    return quaternion(normalize(q.v));
}
CHDI quaternion normalizeSafe(const quaternion& q) {
    return quaternion(normalizeSafe(q.v));
}
// use conjugate instead for rotation-only (normalized) quaternions
CHDI quaternion invert(const quaternion& q) {
    quaternion r = ~q;
    r.v /= lengthSq(r.v);
    return r;
}
// nlerp: minimal path around the sphere, not constant velocity, commutative, faster to compute
// slerp: minimal path around the sphere, constant velocity, not commutative, slow to compute
// nlerp is usually a better choice than slerp
CHDI quaternion nlerp(const quaternion& a, const quaternion& b, float c) {
    float m0 = 1.0f - c;

    quaternion target = b;
    float d = dot(a.v, b.v);
    if (d < 0.0f) { // ensure we're taking the short way around
        target = -b;
    }

    // linearly interpolate and renormalize
    return normalize(quaternion(a.v * m0 + target.v * c));
}

struct transform {
    quaternion rotation;
    vector3 translation;
    float scale;

    CHD transform() : transform(transform::identity()) {}
    CHD transform(const vector3& _translation, const quaternion& _rotation, float _scale)
        : translation(_translation), rotation(_rotation), scale(_scale) {}
    CHD explicit transform(const vector3& _translation) : transform(_translation, quaternion::identity(), 1.0f) {}
    CHD explicit transform(const quaternion& _rotation) : transform(vector3(0.0f), _rotation, 1.0f) {}
    CHD explicit transform(float _scale) : transform(vector3(0.0f), quaternion::identity(), _scale) {}

    CHD static transform identity() {
        return transform(vector3(0.0f), quaternion::identity(), 1.0f);
    }

    // note - this treats v as a direction, and does not apply translation
    CHD vector3 operator*(const vector3& v) const {
        return rotation * (v * scale);
    }
    CHD vector4 operator*(const vector4& v) const {
        return vector4(translation * v.w + rotation * (vector3(v) * scale), v.w);
    }

    CHD transform operator*(const transform& t) const {
        return transform(vector3(*this * vector4(t.translation, 1.0f)), rotation * t.rotation, scale * t.scale);
    }
};

CHDI transform invert(const transform& t) {
    float scale = 1.0f / t.scale;
    quaternion rotation = ~t.rotation;
    vector3 translation = rotation * t.translation * -scale;
    return transform(translation, rotation, scale);
}

// Column-major
struct matrix3x3 {
    vector3 m0, m1, m2;

    CHD matrix3x3() {}
    CHD matrix3x3(vector3 _m0, vector3 _m1, vector3 _m2) : m0(_m0), m1(_m1), m2(_m2) {}
    CHD explicit matrix3x3(const quaternion& q) {
        float x2, y2, z2, xx, yy, zz, xy, yz, xz, wx, wy, wz;
        x2 = q.v.x + q.v.x;
        y2 = q.v.y + q.v.y;
        z2 = q.v.z + q.v.z;
        xx = q.v.x * x2;
        yy = q.v.y * y2;
        zz = q.v.z * z2;
        xy = q.v.x * y2;
        yz = q.v.y * z2;
        xz = q.v.z * x2;
        wx = q.v.w * x2;
        wy = q.v.w * y2;
        wz = q.v.w * z2;

        m0 = vector3(1.0f - (yy + zz), xy + wz, xz - wy);
        m1 = vector3(xy - wz, 1.0f - (xx + zz), yz + wx);
        m2 = vector3(xz + wy, yz - wx, 1.0f - (xx + yy));
    }

    // construct a quaternion from a rotation matrix
    CHD explicit operator quaternion() const {
        float t = 1.0f + m0.x + m1.y + m2.z;
        if (t > .000001f) {
            float s = sqrtf(t) * 2;
            float invS = 1.0f / s;
            return quaternion((m2.y - m1.z) * invS, (m0.z - m2.x) * invS, (m1.x - m0.y) * invS, 0.25f * s);
        } else {
            if (m0.x > m1.y && m0.x > m2.z) {
                float s = sqrtf(1.0f + m0.x - m1.y - m2.z) * 2;
                float invS = 1.0f / s;
                return quaternion(0.25f * s, (m1.x + m0.y) * invS, (m0.z + m2.x) * invS, (m2.y - m1.z) * invS);
            } else if (m1.y > m2.z) {
                float s = sqrtf(1.0f + m1.y - m0.x - m2.z) * 2;
                float invS = 1.0f / s;
                return quaternion((m1.x + m0.y) * invS, 0.25f * s, (m2.y + m1.z) * invS, (m0.z - m2.x) * invS);
            } else {
                float s = sqrtf(1.0f + m2.z - m0.x - m1.y) * 2;
                float invS = 1.0f / s;
                return quaternion((m0.z + m2.x) * invS, (m2.y + m1.z) * invS, 0.25f * s, (m1.x - m0.y) * invS);
            }
        }
    }

    CHD static matrix3x3 zero() {
        return matrix3x3(vector3(0.0f), vector3(0.0f), vector3(0.0f));
    }
    CHD static matrix3x3 identity() {
        return matrix3x3(vector3(1.0f, 0.0f, 0.0f), vector3(0.0f, 1.0f, 0.0f), vector3(0.0f, 0.0f, 1.0f));
    }
    CHD static matrix3x3 scale(float s) {
        return matrix3x3(vector3(s, 0.0f, 0.0f), vector3(0.0f, s, 0.0f), vector3(0.0f, 0.0f, s));
    }
    CHD static matrix3x3 diagonal(float s) {
        return scale(s);
    }
    // look = -z, up = y
    CHD static matrix3x3 camera(const vector3& look, const vector3& up) {
        vector3 z = normalize(-look);
        vector3 y = normalize(up - dot(up, z) * z);
        vector3 x = cross(y, z);
        return matrix3x3(vector3(x), vector3(y), vector3(z));
    }
    // Generates a rotation matrix with z axis align along \param z
    CHD static matrix3x3 rotationFromZAxis(const vector3& z) {
        vector3 y = (fabsf(z.y) > 0.85f) ? vector3(-1.0f, 0.0f, 0.0f) : vector3(0.0f, 1.0f, 0.0f);
        vector3 x = normalize(cross(y, z));
        y = normalize(cross(z, x));
        return matrix3x3(vector3(x), vector3(y), vector3(z));
    }
    // Return the skew-symmetric matrix such that crossProductMatrix(v1)*v2 == cross(v1,v2)
    CHD static matrix3x3 crossProductMatrix(const vector3& v) {
        return matrix3x3(vector3(0.0f, v.z, -v.y), vector3(-v.z, 0, v.x), vector3(v.y, -v.x, 0.0f));
    }
    // Return a matrix which rotates around a unit axis vector.
    CHD static matrix3x3 axisAngle(const vector3& axis, float radians) {
        float c = cosf(radians);
        float s = sinf(radians);
        matrix3x3 k = crossProductMatrix(axis);
        return identity() + k * s + (k * k) * (1.0f - c);
    }

    CHD matrix3x3 operator-() const {
        return matrix3x3(-m0, -m1, -m2);
    }

    CHD matrix3x3 operator+(const matrix3x3& m) const {
        return matrix3x3(m0 + m.m0, m1 + m.m1, m2 + m.m2);
    }
    CHD matrix3x3 operator-(const matrix3x3& m) const {
        return matrix3x3(m0 - m.m0, m1 - m.m1, m2 - m.m2);
    }
    CHD matrix3x3 operator*(const matrix3x3& m) const {
        return matrix3x3(*this * m.m0, *this * m.m1, *this * m.m2);
    }

    CHD matrix3x3 operator*(float s) const {
        return matrix3x3(m0 * s, m1 * s, m2 * s);
    }
    CHD matrix3x3 operator/(float s) const {
        float sInv = 1.0f / s;
        return matrix3x3(m0 * sInv, m1 * sInv, m2 * sInv);
    }

    CHD matrix3x3& operator+=(const matrix3x3& m) {
        return *this = *this + m;
    }
    CHD matrix3x3& operator-=(const matrix3x3& m) {
        return *this = *this - m;
    }
    CHD matrix3x3& operator*=(const matrix3x3& m) {
        return *this = *this * m;
    }

    CHD matrix3x3& operator*=(float s) {
        return *this = *this * s;
    }
    CHD matrix3x3& operator/=(float s) {
        return *this = *this / s;
    }

    CHD vector3 operator*(const vector3& v) const {
        return vector3(m0.x * v.x + m1.x * v.y + m2.x * v.z, m0.y * v.x + m1.y * v.y + m2.y * v.z,
                       m0.z * v.x + m1.z * v.y + m2.z * v.z);
    }
};

CHDI matrix3x3 transpose(const matrix3x3& m) {
    return matrix3x3(vector3(m.m0.x, m.m1.x, m.m2.x), vector3(m.m0.y, m.m1.y, m.m2.y), vector3(m.m0.z, m.m1.z, m.m2.z));
}

CHDI matrix3x3 invert(const matrix3x3& m) {
    const vector3& x = m.m0;
    const vector3& y = m.m1;
    const vector3& z = m.m2;

    float det = x.x * (y.y * z.z - z.y * y.z) - x.y * (y.x * z.z - y.z * z.x) + x.z * (y.x * z.y - y.y * z.x);

    return matrix3x3(vector3(y.y * z.z - z.y * y.z, x.z * z.y - x.y * z.z, x.y * y.z - x.z * y.y),
                     vector3(y.z * z.x - y.x * z.z, x.x * z.z - x.z * z.x, y.x * x.z - x.x * y.z),
                     vector3(y.x * z.y - z.x * y.y, z.x * x.y - x.x * z.y, x.x * y.y - y.x * x.y)) /
           det;
}

struct matrix4x4 {
    vector4 m0, m1, m2, m3;

    CHD matrix4x4() {}
    CHD matrix4x4(vector4 _m0, vector4 _m1, vector4 _m2, vector4 _m3) : m0(_m0), m1(_m1), m2(_m2), m3(_m3) {}
    CHD matrix4x4(const matrix3x3& m, const vector3& translation)
        : m0(m.m0, 0.0f), m1(m.m1, 0.0f), m2(m.m2, 0.0f), m3(translation, 1.0f) {}
    CHD explicit matrix4x4(const matrix3x3& m)
        : m0(m.m0, 0.0f), m1(m.m1, 0.0f), m2(m.m2, 0.0f), m3(0.0f, 0.0f, 0.0f, 1.0f) {}
    CHD explicit matrix4x4(const transform& t) : matrix4x4(matrix3x3(t.rotation) * t.scale, t.translation) {}

    CHD explicit operator matrix3x3() const {
        return matrix3x3(vector3(m0), vector3(m1), vector3(m2));
    }

    // construct a quaternion from a rotation matrix
    CHD explicit operator quaternion() const {
        return quaternion(matrix3x3(*this));
    }

    CHD static matrix4x4 zero() {
        return matrix4x4(vector4(0.0f), vector4(0.0f), vector4(0.0f), vector4(0.0f));
    }
    CHD static matrix4x4 identity() {
        return matrix4x4(vector4(1.0f, 0.0f, 0.0f, 0.0f), vector4(0.0f, 1.0f, 0.0f, 0.0f),
                         vector4(0.0f, 0.0f, 1.0f, 0.0f), vector4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    CHD static matrix4x4 scale(float s) {
        return matrix4x4(vector4(s, 0.0f, 0.0f, 0.0f), vector4(0.0f, s, 0.0f, 0.0f), vector4(0.0f, 0.0f, s, 0.0f),
                         vector4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    CHD static matrix4x4 diagonal(float s) {
        return matrix4x4(vector4(s, 0.0f, 0.0f, 0.0f), vector4(0.0f, s, 0.0f, 0.0f), vector4(0.0f, 0.0f, s, 0.0f),
                         vector4(0.0f, 0.0f, 0.0f, s));
    }
    CHD static matrix4x4 translation(const vector3& t) {
        return matrix4x4(vector4(1.0f, 0.0f, 0.0f, 0.0f), vector4(0.0f, 1.0f, 0.0f, 0.0f),
                         vector4(0.0f, 0.0f, 1.0f, 0.0f), vector4(t, 1.0f));
    }
    // look = -z, up = y
    CHD static matrix4x4 camera(const vector3& look, const vector3& up, const vector3& pos) {
        vector3 z = normalize(-look);
        vector3 y = normalize(up - dot(up, z) * z);
        vector3 x = cross(y, z);
        return matrix4x4(vector4(x), vector4(y), vector4(z), vector4(pos, 1.0f));
    }
    CHD static matrix4x4 lookAt(const vector3& target, const vector3& up, const vector3& pos) {
        vector3 look = target - pos;
        return camera(look, up, pos);
    }
    CHD static matrix4x4 perspectiveProjection(
        float leftTanFov, float rightTanFov, float downTanFov, float upTanFov, float nearZ) {
        float hFovInv = 1.0f / (rightTanFov + leftTanFov);
        float vFovInv = 1.0f / (upTanFov + downTanFov);

        return matrix4x4(vector4(2.0f * hFovInv, 0.0f, 0.0f, 0.0f), vector4(0.0f, 2.0f * vFovInv, 0.0f, 0.0f),
                         vector4((rightTanFov - leftTanFov) * hFovInv, (upTanFov - downTanFov) * vFovInv, 0.0f, -1.0f),
                         vector4(0.0f, 0.0f, nearZ, 0.0f));
    }
    // Return the skew-symmetric matrix such that crossProductMatrix(v1)*v2 == cross(v1,v2)
    CHD static matrix4x4 crossProductMatrix(const vector3& v) {
        return matrix4x4(vector4(0.0f, v.z, -v.y, 0.0f), vector4(-v.z, 0, v.x, 0.0f), vector4(v.y, -v.x, 0.0f, 0.0f),
                         vector4(0.0f));
    }
    // Return a matrix which rotates around a unit axis vector. An optional translation
    // point allows defining a full rigid transform matrix.
    // Note that the t vector is simply used as the 4th column of the result, so it should
    // be a point (x,y,z,1) rather than a vector (x,y,z,0).
    CHD static matrix4x4 axisAngle(const vector3& axis, float radians, const vector3& t = vector3(0.0f)) {
        float c = cosf(radians);
        float s = sinf(radians);
        matrix4x4 k = crossProductMatrix(axis);
        return matrix4x4::translation(t) + k * s + (k * k) * (1.0f - c);
    }
    // Return a matrix which rotates around a pivot point
    CHD static matrix4x4 axisAnglePivot(const vector3& axis, float angle, const vector3& pivot) {
        vector3 npivot = -pivot;
        matrix4x4 r = axisAngle(axis, angle, npivot);
        r.m3 = r * vector4(npivot);
        return r;
    }

    CHD matrix4x4 operator-() const {
        return matrix4x4(-m0, -m1, -m2, -m3);
    }

    CHD matrix4x4 operator+(const matrix4x4& m) const {
        return matrix4x4(m0 + m.m0, m1 + m.m1, m2 + m.m2, m3 + m.m3);
    }
    CHD matrix4x4 operator-(const matrix4x4& m) const {
        return matrix4x4(m0 - m.m0, m1 - m.m1, m2 - m.m2, m3 - m.m3);
    }
    CHD matrix4x4 operator*(const matrix4x4& m) const {
        return matrix4x4(*this * m.m0, *this * m.m1, *this * m.m2, *this * m.m3);
    }

    CHD matrix4x4 operator*(float s) const {
        return matrix4x4(m0 * s, m1 * s, m2 * s, m3 * s);
    }
    CHD matrix4x4 operator/(float s) const {
        float sInv = 1.0f / s;
        return matrix4x4(m0 * sInv, m1 * sInv, m2 * sInv, m3 * sInv);
    }

    CHD matrix4x4& operator+=(const matrix4x4& m) {
        return *this = *this + m;
    }
    CHD matrix4x4& operator-=(const matrix4x4& m) {
        return *this = *this - m;
    }
    CHD matrix4x4& operator*=(const matrix4x4& m) {
        return *this = *this * m;
    }

    CHD matrix4x4& operator*=(float s) {
        return *this = *this * s;
    }
    CHD matrix4x4& operator/=(float s) {
        return *this = *this / s;
    }

    CHD vector4 operator*(const vector4& v) const {
        return vector4(
            m0.x * v.x + m1.x * v.y + m2.x * v.z + m3.x * v.w, m0.y * v.x + m1.y * v.y + m2.y * v.z + m3.y * v.w,
            m0.z * v.x + m1.z * v.y + m2.z * v.z + m3.z * v.w, m0.w * v.x + m1.w * v.y + m2.w * v.z + m3.w * v.w);
    }
};

CHDI matrix4x4 transpose(const matrix4x4& m) {
    return matrix4x4(vector4(m.m0.x, m.m1.x, m.m2.x, m.m3.x), vector4(m.m0.y, m.m1.y, m.m2.y, m.m3.y),
                     vector4(m.m0.z, m.m1.z, m.m2.z, m.m3.z), vector4(m.m0.w, m.m1.w, m.m2.w, m.m3.w));
}

CHDI matrix4x4 invert(const matrix4x4& m) {
    const vector4& x = m.m0;
    const vector4& y = m.m1;
    const vector4& z = m.m2;
    const vector4& w = m.m3;

    float s0 = x.x * y.y - y.x * x.y;
    float s1 = x.x * y.z - y.x * x.z;
    float s2 = x.x * y.w - y.x * x.w;
    float s3 = x.y * y.z - y.y * x.z;
    float s4 = x.y * y.w - y.y * x.w;
    float s5 = x.z * y.w - y.z * x.w;
    float c5 = z.z * w.w - w.z * z.w;
    float c4 = z.y * w.w - w.y * z.w;
    float m3 = z.y * w.z - w.y * z.z;
    float m2 = z.x * w.w - w.x * z.w;
    float m1 = z.x * w.z - w.x * z.z;
    float m0 = z.x * w.y - w.x * z.y;
    float det = s0 * c5 - s1 * c4 + s2 * m3 + s3 * m2 - s4 * m1 + s5 * m0;

    float idet = 1.0f / det;
    s0 *= idet;
    s1 *= idet;
    s2 *= idet;
    s3 *= idet;
    s4 *= idet;
    s5 *= idet;
    m0 *= idet;
    m1 *= idet;
    m2 *= idet;
    m3 *= idet;
    c4 *= idet;
    c5 *= idet;

    return matrix4x4(vector4(y.y * c5 - y.z * c4 + y.w * m3, x.z * c4 - x.y * c5 - x.w * m3,
                             w.y * s5 - w.z * s4 + w.w * s3, z.z * s4 - z.y * s5 - z.w * s3),
                     vector4(y.z * m2 - y.x * c5 - y.w * m1, x.x * c5 - x.z * m2 + x.w * m1,
                             w.z * s2 - w.x * s5 - w.w * s1, z.x * s5 - z.z * s2 + z.w * s1),
                     vector4(y.x * c4 - y.y * m2 + y.w * m0, x.y * m2 - x.x * c4 - x.w * m0,
                             w.x * s4 - w.y * s2 + w.w * s0, z.y * s2 - z.x * s4 - z.w * s0),
                     vector4(y.y * m1 - y.x * m3 - y.z * m0, x.x * m3 - x.y * m1 + x.z * m0,
                             w.y * s1 - w.x * s3 - w.z * s0, z.x * s3 - z.y * s1 + z.z * s0));
}

// Return the inverse of matrix m, which must represent a rigid transformation:
// the upper 3x3 submatrix of m must be orthogonal, and the bottom row must be
// (0,0,0,1) or (0,0,0,0). In the latter case the matrix has no inverse, but
// this will still return the pseudoinverse.
CHDI matrix4x4 invertRigid(const matrix4x4& m) {
    matrix4x4 r(vector4(m.m0.x, m.m1.x, m.m2.x, 0.0f), vector4(m.m0.y, m.m1.y, m.m2.y, 0.0f),
                vector4(m.m0.z, m.m1.z, m.m2.z, 0.0f), vector4(0.0f, 0.0f, 0.0f, -1.0f));
    r.m3 = r * -m.m3;
    return r;
}

} // namespace hvvr
