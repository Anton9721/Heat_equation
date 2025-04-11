#pragma once
#include "Array2D.hpp"
#include <vector>
#include <span>

class Solver {
public:
    virtual void solve(Array2D& temp_data, std::span<const double> initial_conditions,
        double a, int n_x, int n_t, double length, double t_max) const = 0;
    virtual ~Solver() = default;
};

class Solver_Euler : public Solver {
public:
    void solve(Array2D& temp_data, std::span<const double> initial_conditions,
              double a, int n_x, int n_t, double length, double t_max) const override;
};

class Solver_Crank_Nicolson : public Solver {
public:
    void solve(Array2D& temp_data, std::span<const double> initial_conditions,
              double a, int n_x, int n_t, double length, double t_max) const override;
};

class Solver_Crank_Nicolson_modified{
    public:
        void solve(Array2D& temp_data, std::span<const double> initial_conditions, 
            double a, const int n_x, int n_t, 
            double length, double t_max, double h, double u_c) const;
};