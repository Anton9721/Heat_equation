#pragma once

#include "Array2D.hpp"
#include "Writer.hpp"
#include <span>
#include <vector>
#include <cmath>
#include <numbers>
#include <omp.h>


class Solver{
public:
    virtual void solve(Array2D<double> &temp_data, std::vector<double> initial_conditions, 
        const double a, const int n_x, const int n_t, const double length, const double t_max) const = 0;
    virtual ~Solver() = default;          

};

//Решение методом Эйлера
class Solver_Euler : public Solver
{
    public:
        void solve(Array2D<double> &temp_data, std::vector<double> initial_conditions, 
                  const double a, const int n_x, const int n_t, const double length, const double t_max) const override;
};


//Решение методом Кранка-Николсона
class Solver_Сrank_Nicolson : public Solver
{
    public:
        void solve(Array2D<double> &temp_data, std::vector<double> initial_conditions, 
                  const double a, const int n_x, const int n_t, const double length, const double t_max) const override;
};

//Решение методом Кранка-Николсона (Охлаждение Ньютона)
class Solver_Crank_Nicolson_modified{
    public:
        void solve(Array2D<double> &temp_data, std::vector<double> initial_conditions, 
            const double a, const int n_x, const int n_t, 
            const double length, const double t_max, const double h, const double u_c) const;
};