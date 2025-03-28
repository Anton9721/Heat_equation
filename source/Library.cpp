# include "Library.hpp"



std::vector<double> data_temp(std::string solver_name, 
    std::vector<double> initial_conditions,
    double a, int n_x, int n_t, 
    double length, double t_max, 
    double h, double u_c)
{
    Array2D<double> temp_data(n_t, n_x);
    std::vector<double> data_result;

    if (solver_name == "Euler_method")
    {
        Solver_Euler solver;
        solver.solve(temp_data, initial_conditions, a, n_x, n_t, length, t_max);

        for (size_t i_t = 0; i_t < n_t; ++i_t) {
            for (size_t i_x = 0; i_x < n_x; ++i_x) {
                data_result.push_back(temp_data(i_t, i_x));
                }
            }
    }

    else if (solver_name == "Nicholson_Crunk_method")
    {
        Solver_Сrank_Nicolson solver;
        solver.solve(temp_data, initial_conditions, a, n_x, n_t, length, t_max);

        for (size_t i_t = 0; i_t < n_t; ++i_t) {
            for (size_t i_x = 0; i_x < n_x; ++i_x) {
                data_result.push_back(temp_data(i_t, i_x));
                }
            }
    }


    else if (solver_name == "Nicholson_Crunk_method_modified")
    {
        Solver_Crank_Nicolson_modified solver;
        solver.solve(temp_data, initial_conditions, a, n_x, n_t, length, t_max, h, u_c);

        for (size_t i_t = 0; i_t < n_t; ++i_t) {
            for (size_t i_x = 0; i_x < n_x; ++i_x) {
                data_result.push_back(temp_data(i_t, i_x));
                }
            }
    }

    return data_result;
}

