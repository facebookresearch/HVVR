#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>
#include <xmmintrin.h>

namespace hvvr {

//==============================================================================
// __m128 overloads
//==============================================================================

__forceinline __m128 m128(float x, float y, float z = 0, float w = 0) { return _mm_set_ps(w, z, y, x); }
__forceinline __m128 m128(float a) { return _mm_set_ps1(a); }
__forceinline __m128 m128(__m128i a) { return _mm_castsi128_ps(a); }
__forceinline __m128 m128(__m256 a) { return _mm256_castps256_ps128(a); }
__forceinline __m128 load_m128(const float* p) { return _mm_load_ps(p); }
__forceinline __m128 loadu_m128(const float* p) { return _mm_load_ps(p); }
__forceinline __m128 broadcast_m128(const float* p) { return _mm_broadcast_ss(p); }
__forceinline void store(float* p, __m128 a) { _mm_store_ps(p, a); }
__forceinline void storeu(float* p, __m128 a) { _mm_storeu_ps(p, a); }
__forceinline float m128_get(const __m128& a, size_t i) { return a.m128_f32[i]; }
__forceinline __m128 operator-(__m128 a) { return _mm_xor_ps(a, m128(-0.0f)); }
__forceinline __m128 operator+(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
__forceinline __m128 operator-(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
__forceinline __m128 operator*(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
__forceinline __m128 operator/(__m128 a, __m128 b) { return _mm_div_ps(a, b); }
__forceinline __m128 operator*(__m128 a, float b) { return a * m128(b); }
__forceinline __m128 operator/(__m128 a, float b) { return a / m128(b); }
__forceinline __m128 operator*(float a, __m128 b) { return m128(a) * b; }
__forceinline __m128 operator/(float a, __m128 b) { return m128(a) / b; }
__forceinline __m128 operator&(__m128 a, __m128 b) { return _mm_and_ps(a, b); }
__forceinline __m128 operator|(__m128 a, __m128 b) { return _mm_or_ps(a, b); }
__forceinline __m128 operator^(__m128 a, __m128 b) { return _mm_xor_ps(a, b); }
__forceinline __m128 operator<(__m128 a, __m128 b) { return _mm_cmplt_ps(a, b); }
__forceinline __m128 operator>(__m128 a, __m128 b) { return _mm_cmpgt_ps(a, b); }
__forceinline __m128 operator<=(__m128 a, __m128 b) { return _mm_cmple_ps(a, b); }
__forceinline __m128 operator>=(__m128 a, __m128 b) { return _mm_cmpge_ps(a, b); }
__forceinline __m128 operator==(__m128 a, __m128 b) { return _mm_cmpeq_ps(a, b); }
__forceinline __m128 operator!=(__m128 a, __m128 b) { return _mm_cmpneq_ps(a, b); }
__forceinline __m128& operator+=(__m128& a, __m128 b) { return a = a + b; }
__forceinline __m128& operator-=(__m128& a, __m128 b) { return a = a - b; }
__forceinline __m128& operator*=(__m128& a, __m128 b) { return a = a * b; }
__forceinline __m128& operator/=(__m128& a, __m128 b) { return a = a / b; }
__forceinline __m128& operator*=(__m128& a, float b) { return a = a * b; }
__forceinline __m128& operator/=(__m128& a, float b) { return a = a / b; }
#ifdef __AVX2__
__forceinline __m128 fmadd(__m128 a, __m128 b, __m128 c) { return _mm_fmadd_ps(a, b, c); }
__forceinline __m128 fmsub(__m128 a, __m128 b, __m128 c) { return _mm_fmsub_ps(a, b, c); }
__forceinline __m128 fnmadd(__m128 a, __m128 b, __m128 c) { return _mm_fnmadd_ps(a, b, c); }
__forceinline __m128 fnmsub(__m128 a, __m128 b, __m128 c) { return _mm_fnmsub_ps(a, b, c); }
#else
__forceinline __m128 fmadd(__m128 a, __m128 b, __m128 c) { return a*b + c; }
__forceinline __m128 fmsub(__m128 a, __m128 b, __m128 c) { return a*b - c; }
__forceinline __m128 fnmadd(__m128 a, __m128 b, __m128 c) { return -(a*b) + c; }
__forceinline __m128 fnmsub(__m128 a, __m128 b, __m128 c) { return -(a*b) - c; }
#endif
__forceinline __m128 min(__m128 a, __m128 b) { return _mm_min_ps(a, b); }
__forceinline __m128 max(__m128 a, __m128 b) { return _mm_max_ps(a, b); }
__forceinline __m128 sqrt(__m128 a) { return _mm_sqrt_ps(a); }

__forceinline __m128 rcp_nr(__m128 a) {
    __m128 r = _mm_rcp_ps(a);
    return (m128(2) - a * r) * r;
}
__forceinline __m128 rsqrt_nr(__m128 a) {
    __m128 r = _mm_rsqrt_ps(a);
    return m128(0.5f) * r * (m128(3) - a * r * r);
}

__forceinline unsigned movemask(__m128 a) { return (unsigned)_mm_movemask_ps(a); }

template <unsigned i> __forceinline __m128 extract_m128(__m256 a) {
    static_assert(i < 2, "invalid extract index");
    return _mm256_extractf128_ps(a, i);
}
template <unsigned x, unsigned y, unsigned z, unsigned w> __forceinline __m128 shuffle(__m128 a) {
    static_assert(x < 4 && y < 4 && z < 4 && w < 4, "invalid shuffle index");
    return m128(_mm_shuffle_epi32(_mm_castps_si128(a), w * 64 + z * 16 + y * 4 + x));
}
template <unsigned x, unsigned y, unsigned z, unsigned w> __forceinline __m128 shuffle(__m128 a, __m128 b) {
    static_assert(x < 4 && y < 4 && z < 4 && w < 4, "invalid shuffle index");
    return _mm_shuffle_ps(a, b, w * 64 + z * 16 + y * 4 + x);
}
template <unsigned x, unsigned y, unsigned z, unsigned w> __forceinline __m128 blend(__m128 a, __m128 b) {
    static_assert(x < 2 && y < 2 && z < 2 && w < 2, "invalid blend index");
    return _mm_blend_ps(a, b, w * 8 + z * 4 + y * 2 + x);
}
template <unsigned i> __forceinline __m128 broadcast(__m128 a) { return shuffle<i, i, i, i>(a); }
__forceinline __m128 unpacklo(__m128 a, __m128 b) { return _mm_unpacklo_ps(a, b); }
__forceinline __m128 unpackhi(__m128 a, __m128 b) { return _mm_unpackhi_ps(a, b); }

//==============================================================================
// __m256 overloads
//==============================================================================

__forceinline __m256 m256(__m128 a) { return _mm256_castps128_ps256(a); }
__forceinline __m256 m256(__m128 a, __m128 b) { return _mm256_insertf128_ps(m256(a), b, 1); }
__forceinline __m256 m256(float a, float b, float c, float d, float e, float f, float g, float h) {
    return _mm256_set_ps(h, g, f, e, d, c, b, a);
}
__forceinline __m256 m256(float a) { return _mm256_set1_ps(a); }
__forceinline __m256 m256(__m256i a) { return _mm256_castsi256_ps(a); }
__forceinline __m256 load_m256(const float* p) { return _mm256_load_ps(p); }
__forceinline __m256 loadu_m256(const float* p) { return _mm256_load_ps(p); }
__forceinline __m256 broadcast_m256(const float* p) { return _mm256_broadcast_ss(p); }
__forceinline __m256 broadcast_m256(const __m128* p) { return _mm256_broadcast_ps(p); }
__forceinline void store(float* p, __m256 a) { _mm256_store_ps(p, a); }
__forceinline void storeu(float* p, __m256 a) { _mm256_storeu_ps(p, a); }
__forceinline float m256_get(const __m256& a, size_t i) { return a.m256_f32[i]; }
__forceinline __m256 operator-(__m256 a) { return _mm256_xor_ps(a, m256(-0.0f)); }
__forceinline __m256 operator&(__m256 a, __m256 b) { return _mm256_and_ps(a, b); }
__forceinline __m256 operator|(__m256 a, __m256 b) { return _mm256_or_ps(a, b); }
__forceinline __m256 operator^(__m256 a, __m256 b) { return _mm256_xor_ps(a, b); }
__forceinline __m256 operator+(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
__forceinline __m256 operator-(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
__forceinline __m256 operator*(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
__forceinline __m256 operator/(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
__forceinline __m256 operator*(__m256 a, float b) { return a * m256(b); }
__forceinline __m256 operator/(__m256 a, float b) { return a / m256(b); }
__forceinline __m256 operator*(float a, __m256 b) { return m256(a) * b; }
__forceinline __m256 operator/(float a, __m256 b) { return m256(a) / b; }
__forceinline __m256 operator<(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
__forceinline __m256 operator>(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
__forceinline __m256 operator<=(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
__forceinline __m256 operator>=(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
__forceinline __m256 operator==(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
__forceinline __m256 operator!=(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ); }
__forceinline __m256& operator&=(__m256& a, __m256 b) { return a = a & b; }
__forceinline __m256& operator|=(__m256& a, __m256 b) { return a = a | b; }
__forceinline __m256& operator^=(__m256& a, __m256 b) { return a = a ^ b; }
__forceinline __m256& operator+=(__m256& a, __m256 b) { return a = a + b; }
__forceinline __m256& operator-=(__m256& a, __m256 b) { return a = a - b; }
__forceinline __m256& operator*=(__m256& a, __m256 b) { return a = a * b; }
__forceinline __m256& operator/=(__m256& a, __m256 b) { return a = a / b; }
__forceinline __m256& operator*=(__m256& a, float b) { return a = a * b; }
__forceinline __m256& operator/=(__m256& a, float b) { return a = a / b; }
#ifdef __AVX2__
__forceinline __m256 fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }
__forceinline __m256 fmsub(__m256 a, __m256 b, __m256 c) { return _mm256_fmsub_ps(a, b, c); }
__forceinline __m256 fnmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fnmadd_ps(a, b, c); }
__forceinline __m256 fnmsub(__m256 a, __m256 b, __m256 c) { return _mm256_fnmsub_ps(a, b, c); }
#else
__forceinline __m256 fmadd(__m256 a, __m256 b, __m256 c) { return a*b + c; }
__forceinline __m256 fmsub(__m256 a, __m256 b, __m256 c) { return a*b - c; }
__forceinline __m256 fnmadd(__m256 a, __m256 b, __m256 c) { return -(a*b) + c; }
__forceinline __m256 fnmsub(__m256 a, __m256 b, __m256 c) { return -(a*b) - c; }
#endif
__forceinline __m256 min(__m256 a, __m256 b) { return _mm256_min_ps(a, b); }
__forceinline __m256 max(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }
__forceinline __m256 sqrt(__m256 a) { return _mm256_sqrt_ps(a); }
__forceinline __m256 rcp_nr(__m256 a) {
    __m256 r = _mm256_rcp_ps(a);
    return fnmadd(a, r, m256(2)) * r;
}
__forceinline __m256 rsqrt_nr(__m256 a) {
    __m256 r = _mm256_rsqrt_ps(a);
    return m256(0.5f) * r * fnmadd(a * r, r, m256(3));
}

__forceinline unsigned movemask(__m256 a) { return (unsigned)_mm256_movemask_ps(a); }

template <unsigned x, unsigned y, unsigned z, unsigned w> __forceinline __m256 shuffle(__m256 a) {
    static_assert(x < 4 && y < 4 && z < 4 && w < 4, "invalid index");
    return m256(_mm256_shuffle_epi32(_mm256_castps_si256(a), w * 64 + z * 16 + y * 4 + x));
}
template <unsigned x, unsigned y, unsigned z, unsigned w> __forceinline __m256 shuffle(__m256 a, __m256 b) {
    static_assert(x < 4 && y < 4 && z < 4 && w < 4, "invalid index");
    return _mm256_shuffle_ps(a, b, w * 64 + z * 16 + y * 4 + x);
}
template <unsigned i0, unsigned i1, unsigned i2, unsigned i3, unsigned i4, unsigned i5, unsigned i6, unsigned i7>
__forceinline __m256 shuffle(__m256 a) {
    static_assert(i0 < 8 && i1 < 8 && i2 < 8 && i3 < 8 && i4 < 8 && i5 < 8 && i6 < 8 && i7 < 8, "invalid index");
    static const __m256i perm = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
    return _mm256_permutevar8x32_ps(a, perm);
}
__forceinline __m256 unpacklo(__m256 a, __m256 b) { return _mm256_unpacklo_ps(a, b); }
__forceinline __m256 unpackhi(__m256 a, __m256 b) { return _mm256_unpackhi_ps(a, b); }

//==============================================================================
// __m128d overloads
//==============================================================================

__forceinline __m128d m128d(double x, double y) { return _mm_set_pd(y, x); }
__forceinline __m128d m128d(double a) { return _mm_set1_pd(a); }
__forceinline __m128d m128d(__m128i a) { return _mm_castsi128_pd(a); }
__forceinline __m128d m128d(__m256d a) { return _mm256_castpd256_pd128(a); }
__forceinline __m128d load_m128d(const double* p) { return _mm_load_pd(p); }
__forceinline __m128d loadu_m128d(const double* p) { return _mm_load_pd(p); }
__forceinline __m128d broadcast_m128d(const double* p) { return m128d(_mm256_broadcast_sd(p)); }
__forceinline void store(double* p, __m128d a) { _mm_store_pd(p, a); }
__forceinline void storeu(double* p, __m128d a) { _mm_storeu_pd(p, a); }
__forceinline double m256d_get(const __m256d& a, size_t i) { return a.m256d_f64[i]; }
__forceinline __m128d operator-(__m128d a) { return _mm_xor_pd(a, m128d(-0.0f)); }
__forceinline __m128d operator&(__m128d a, __m128d b) { return _mm_and_pd(a, b); }
__forceinline __m128d operator|(__m128d a, __m128d b) { return _mm_or_pd(a, b); }
__forceinline __m128d operator^(__m128d a, __m128d b) { return _mm_xor_pd(a, b); }
__forceinline __m128d operator+(__m128d a, __m128d b) { return _mm_add_pd(a, b); }
__forceinline __m128d operator-(__m128d a, __m128d b) { return _mm_sub_pd(a, b); }
__forceinline __m128d operator*(__m128d a, __m128d b) { return _mm_mul_pd(a, b); }
__forceinline __m128d operator/(__m128d a, __m128d b) { return _mm_div_pd(a, b); }
__forceinline __m128d operator*(__m128d a, double b) { return a * m128d(b); }
__forceinline __m128d operator/(__m128d a, double b) { return a / m128d(b); }
__forceinline __m128d operator*(double a, __m128d b) { return m128d(a) * b; }
__forceinline __m128d operator/(double a, __m128d b) { return m128d(a) / b; }
__forceinline __m128d operator<(__m128d a, __m128d b) { return _mm_cmplt_pd(a, b); }
__forceinline __m128d operator>(__m128d a, __m128d b) { return _mm_cmpgt_pd(a, b); }
__forceinline __m128d operator<=(__m128d a, __m128d b) { return _mm_cmple_pd(a, b); }
__forceinline __m128d operator>=(__m128d a, __m128d b) { return _mm_cmpge_pd(a, b); }
__forceinline __m128d operator==(__m128d a, __m128d b) { return _mm_cmpeq_pd(a, b); }
__forceinline __m128d operator!=(__m128d a, __m128d b) { return _mm_cmpneq_pd(a, b); }
__forceinline __m128d& operator&=(__m128d& a, __m128d b) { return a = a & b; }
__forceinline __m128d& operator|=(__m128d& a, __m128d b) { return a = a | b; }
__forceinline __m128d& operator^=(__m128d& a, __m128d b) { return a = a ^ b; }
__forceinline __m128d& operator+=(__m128d& a, __m128d b) { return a = a + b; }
__forceinline __m128d& operator-=(__m128d& a, __m128d b) { return a = a - b; }
__forceinline __m128d& operator*=(__m128d& a, __m128d b) { return a = a * b; }
__forceinline __m128d& operator/=(__m128d& a, __m128d b) { return a = a / b; }
__forceinline __m128d& operator*=(__m128d& a, double b) { return a = a * b; }
__forceinline __m128d& operator/=(__m128d& a, double b) { return a = a / b; }
__forceinline __m128d fmadd(__m128d a, __m128d b, __m128d c) { return _mm_fmadd_pd(a, b, c); }
__forceinline __m128d fmsub(__m128d a, __m128d b, __m128d c) { return _mm_fmsub_pd(a, b, c); }
__forceinline __m128d fnmadd(__m128d a, __m128d b, __m128d c) { return _mm_fnmadd_pd(a, b, c); }
__forceinline __m128d fnmsub(__m128d a, __m128d b, __m128d c) { return _mm_fnmsub_pd(a, b, c); }
__forceinline __m128d min(__m128d a, __m128d b) { return _mm_min_pd(a, b); }
__forceinline __m128d max(__m128d a, __m128d b) { return _mm_max_pd(a, b); }
__forceinline __m128d sqrt(__m128d a) { return _mm_sqrt_pd(a); }

__forceinline unsigned movemask(__m128d a) { return (unsigned)_mm_movemask_pd(a); }

template <unsigned i> __forceinline __m128d extract_m128d(__m256d a) {
	static_assert(i < 2, "invalid index");
	return _mm256_extractf128d_pd(a, i);
}
template <unsigned x, unsigned y> __forceinline __m128d shuffle(__m128d a) {
	static_assert(x < 2 && y < 2, "invalid index");
	return m128d(_mm_shuffle_epi32(_mm_castps_si128d(a), w * 64 + z * 16 + y * 4 + x));
}
template <unsigned x, unsigned y, unsigned z, unsigned w> __forceinline __m128d shuffle(__m128d a, __m128d b) {
	static_assert(x < 4 && y < 4 && z < 4 && w < 4, "invalid index");
	return _mm_shuffle_pd(a, w * 64 + z * 16 + y * 4 + x);
}
template <unsigned i> __forceinline __m128d broadcast(__m128d a) { return shuffle<i, i, i, i>(a); }
__forceinline __m128d unpacklo(__m128d a, __m128d b) { return _mm_unpacklo_pd(a, b); }
__forceinline __m128d unpackhi(__m128d a, __m128d b) { return _mm_unpackhi_pd(a, b); }

//==============================================================================
// __m256d overloads
//==============================================================================

__forceinline __m256d m256d(__m128d a) { return _mm256_castpd128_pd256(a); }
__forceinline __m256d m256d(__m128d a, __m128d b) { return _mm256_insertf128_pd(m256d(a), b, 1); }
__forceinline __m256d m256d(double a, double b, double c, double d) { return _mm256_set_pd(d, c, b, a); }
__forceinline __m256d m256d(double a) { return _mm256_set1_pd(a); }
__forceinline __m256d m256d(__m256i a) { return _mm256_castsi256_pd(a); }
__forceinline __m256d load_m256d(const double* p) { return _mm256_load_pd(p); }
__forceinline __m256d loadu_m256d(const double* p) { return _mm256_load_pd(p); }
__forceinline __m256d broadcast_m256d(const double* p) { return _mm256_broadcast_sd(p); }
__forceinline __m256d broadcast_m256d(const __m128d* p) { return _mm256_broadcast_pd(p); }
__forceinline void store(double* p, __m256d a) { _mm256_store_pd(p, a); }
__forceinline void storeu(double* p, __m256d a) { _mm256_storeu_pd(p, a); }
__forceinline __m256d operator-(__m256d a) { return _mm256_xor_pd(a, m256d(-0.0f)); }
__forceinline __m256d operator&(__m256d a, __m256d b) { return _mm256_and_pd(a, b); }
__forceinline __m256d operator|(__m256d a, __m256d b) { return _mm256_or_pd(a, b); }
__forceinline __m256d operator^(__m256d a, __m256d b) { return _mm256_xor_pd(a, b); }
__forceinline __m256d operator+(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
__forceinline __m256d operator-(__m256d a, __m256d b) { return _mm256_sub_pd(a, b); }
__forceinline __m256d operator*(__m256d a, __m256d b) { return _mm256_mul_pd(a, b); }
__forceinline __m256d operator/(__m256d a, __m256d b) { return _mm256_div_pd(a, b); }
__forceinline __m256d operator*(__m256d a, double b) { return a * m256d(b); }
__forceinline __m256d operator/(__m256d a, double b) { return a / m256d(b); }
__forceinline __m256d operator*(double a, __m256d b) { return m256d(a) * b; }
__forceinline __m256d operator/(double a, __m256d b) { return m256d(a) / b; }
__forceinline __m256d operator<(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
__forceinline __m256d operator>(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
__forceinline __m256d operator<=(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
__forceinline __m256d operator>=(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); }
__forceinline __m256d operator==(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
__forceinline __m256d operator!=(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ); }
__forceinline __m256d& operator&=(__m256d& a, __m256d b) { return a = a & b; }
__forceinline __m256d& operator|=(__m256d& a, __m256d b) { return a = a | b; }
__forceinline __m256d& operator^=(__m256d& a, __m256d b) { return a = a ^ b; }
__forceinline __m256d& operator+=(__m256d& a, __m256d b) { return a = a + b; }
__forceinline __m256d& operator-=(__m256d& a, __m256d b) { return a = a - b; }
__forceinline __m256d& operator*=(__m256d& a, __m256d b) { return a = a * b; }
__forceinline __m256d& operator/=(__m256d& a, __m256d b) { return a = a / b; }
__forceinline __m256d& operator*=(__m256d& a, double b) { return a = a * b; }
__forceinline __m256d& operator/=(__m256d& a, double b) { return a = a / b; }
__forceinline __m256d fmadd(__m256d a, __m256d b, __m256d c) { return _mm256_fmadd_pd(a, b, c); }
__forceinline __m256d fmsub(__m256d a, __m256d b, __m256d c) { return _mm256_fmsub_pd(a, b, c); }
__forceinline __m256d fnmadd(__m256d a, __m256d b, __m256d c) { return _mm256_fnmadd_pd(a, b, c); }
__forceinline __m256d fnmsub(__m256d a, __m256d b, __m256d c) { return _mm256_fnmsub_pd(a, b, c); }
__forceinline __m256d min(__m256d a, __m256d b) { return _mm256_min_pd(a, b); }
__forceinline __m256d max(__m256d a, __m256d b) { return _mm256_max_pd(a, b); }
__forceinline __m256d sqrt(__m256d a) { return _mm256_sqrt_pd(a); }

__forceinline unsigned movemask(__m256d a) { return (unsigned)_mm256_movemask_pd(a); }

template <unsigned x, unsigned y, unsigned z, unsigned w> __forceinline __m256d shuffle(__m256d a) {
	static_assert(x < 4 && y < 4 && z < 4 && w < 4, "invalid index");
	return _mm256_permute4x64_pd(a, w * 64 + z * 16 + y * 4 + x);
}

__forceinline __m256d unpacklo(__m256d a, __m256d b) { return _mm256_unpacklo_pd(a, b); }
__forceinline __m256d unpackhi(__m256d a, __m256d b) { return _mm256_unpackhi_pd(a, b); }

} // namespace hvvr
