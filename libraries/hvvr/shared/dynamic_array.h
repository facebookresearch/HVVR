#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "array_view.h"
#include "util.h"

#include <type_traits>

namespace hvvr {

template <typename T>
class DynamicArray {
public:
    typedef T value_type;
    typedef typename std::remove_const<T>::type mutable_value_type;

    DynamicArray() : _data(nullptr), _size(0) {}
    DynamicArray(const DynamicArray& a) : DynamicArray(a._size) {
        for (size_t i = 0, e = _size; i != e; ++i)
            _data[i] = a._data[i];
    }
    DynamicArray(DynamicArray&& a) : DynamicArray() {
        swap(a);
    }

    explicit DynamicArray(size_t size) : DynamicArray(size, ALIGNOF(T)) {}
    explicit DynamicArray(size_t size, size_t alignment)
        : _data(size ? (T*)_aligned_malloc(sizeof(T) * size, alignment) : nullptr), _size(size) {
        if (!std::is_pod<T>::value)
            for (size_t i = 0; i < _size; ++i)
                new (&_data[i]) T();
    }

    operator ArrayView<const T>() const {
        return ArrayView<const T>{_data, _size};
    }
    operator ArrayView<T>() {
        return ArrayView<T>{_data, _size};
    }

    DynamicArray& operator=(DynamicArray a) {
        swap(a);
        return *this;
    }

    ~DynamicArray() {
        if (_data) {
            if (!std::is_pod<T>::value)
                for (size_t i = 0; i < _size; ++i)
                    _data[i].~T();
            _aligned_free(_data);
        }
    }

    void swap(DynamicArray& a) {
        std::swap(_data, a._data);
        std::swap(_size, a._size);
    }

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

    ArrayView<T> view(size_t first, size_t bound) {
        return ArrayView<T>{_data + first, bound - first};
    }

protected:
    T* _data;
    size_t _size;
};

} // namespace hvvr
