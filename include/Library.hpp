#pragma once

#include "Solver.hpp"
#include "Array2D.hpp"

#include <omp.h>
#include <vector>
#include <string>
#include <span>

void data_temp(
    int solver_name,
    double* result_data,
    const double* initial_conditions,
    double a, int n_x, int n_t,
    double length, double t_max,
    double h, double u_c);