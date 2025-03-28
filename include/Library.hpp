#pragma once

#include "Solver.hpp"
#include "Array2D.hpp"

#include <omp.h>
#include <vector>
#include <string>
#include <span>

std::vector<double> data_temp(std::string solver_name, std::vector<double> initial_conditions, 
    const double a, const int n_x, const int n_t, const double length, const double t_max, const double h, const double u_c);



