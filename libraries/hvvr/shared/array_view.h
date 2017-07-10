#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <array>
#include <vector>

namespace hvvr {

/* ArrayView<T> represents a non-owning reference to a contiguous block of T's,
 * equivalent to just passing a pointer and a count but more convenient, since
 * it looks like a standard container and has implicit conversions from
 * std::vector and std::array.
 *
 * For const data, use `ArrayView<const T>` rather than `const ArrayView<T>`.
 * The former allows you to change where the slice points, but not the pointed-at
 * data. The latter allows you to change the data but not the slice itself.
 * This is analagous to the difference between pointer-to-const and const-pointer.
 */
template <typename T>
class ArrayView {
public:
    typedef T value_type;
    typedef typename std::remove_const<T>::type mutable_value_type;

    ArrayView() : _data(nullptr), _size(0) {}
    ArrayView(T* ptr, size_t n) : _data(ptr), _size(n) {}

    // Implicit constructors from array-like objects.
    ArrayView(const std::vector<mutable_value_type>& v) : _data(v.data()), _size(v.size()) {}
    template <int N>
    ArrayView(std::array<T, N>& v) : _data(v.data()), _size(v.size()) {}
    template <int N>
    ArrayView(const std::array<mutable_value_type, N>& v) : _data(v.data()), _size(v.size()) {}
    template <int N>
    ArrayView(T const(&v)[N]) : _data(v), _size(N) {}

    // Accessors, named for compatibility with std::vector and std::array
    const T& operator[](size_t i) const {
        return _data[i];
    }
    T& operator[](size_t i) {
        return _data[i];
    }
    T* begin() const {
        return _data;
    }
    T* end() const {
        return _data + _size;
    }
    const T* cbegin() const {
        return _data;
    }
    const T* cend() const {
        return _data + _size;
    }
    T* data() const {
        return _data;
    }
    size_t size() const {
        return _size;
    }

    // Return a std::vector containing a copy of this ArrayView's data
    explicit operator std::vector<mutable_value_type>() const {
        return {_data, _data + _size};
    }

    // Return a subrange of this ArrayView, clamped to stay in range.
    ArrayView<T> slice(size_t start, size_t n) const {
        if (start <= _size) {
            return ArrayView(_data + start, std::min(n, _size - start));
        } else {
            return ArrayView();
        }
    }

protected:
    T* _data;
    size_t _size;
};

} // namespace hvvr
