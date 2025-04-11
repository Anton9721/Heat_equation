#include "Library.hpp"
#include "Array2D.hpp"
#include "Solver.hpp"

void data_temp(
    int solver_num,
    double* result_data,
    const double* initial_conditions,
    double a, int n_x, int n_t,
    double length, double t_max,
    double h, double u_c)
{
    Array2D temp_data(n_t, n_x, result_data);
    std::span<const double> init_span(initial_conditions, n_x);

    if (solver_num == 0) {
        Solver_Euler solver;
        solver.solve(temp_data, init_span, a, n_x, n_t, length, t_max);       
    }
    else if (solver_num == 1) {
        Solver_Euler solver;
        solver.solve(temp_data, init_span, a, n_x, n_t, length, t_max);
    }
    else if (solver_num == 2) {
        Solver_Crank_Nicolson_modified solver;
        solver.solve(temp_data, init_span, a, n_x, n_t, length, t_max, h, u_c);
    }
}


