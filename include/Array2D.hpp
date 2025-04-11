#pragma once

#include <iostream>
#include <span>
#include <iomanip>  
#include <omp.h>


class Array2D {
private:
    size_t rows;
    size_t cols;
    std::span<double> data;

public:
    Array2D(size_t r, size_t c, double* ptr) 
        : rows(r), cols(c), data(ptr, r * c) {}

    double& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    std::span<double> getRow(size_t r) {
        return data.subspan(r * cols, cols);
    }
};