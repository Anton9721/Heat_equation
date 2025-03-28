#pragma once

#include <iostream>
#include <vector>
#include <span>
#include <iomanip>  
#include <omp.h>

template <typename T>
class Array2D {
private:
    size_t rows;
    size_t cols;
    std::vector<T> data;

public:
    Array2D(size_t r, size_t c) : rows(r), cols(c), data(r * c) {
        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] = 0.0;
        }
    }

    T &operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    const T &operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};