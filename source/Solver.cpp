#include "Solver.hpp"

void Solver_Euler::solve(Array2D<double> &temp_data, std::vector<double> initial_conditions, 
    const double a, const int n_x, const int n_t, const double length, const double t_max) const
{
    double dx = length / (n_x - 1);
    double dt = t_max / (n_t - 1);
    double r = (a * a * dt) / (dx * dx);

    for (size_t i = 0; i < n_x; ++i) {
        temp_data(0, i) = initial_conditions[i];
    }

    for (size_t t = 0; t < n_t - 1; ++t) {
        #pragma omp parallel for
        for (size_t x = 1; x < n_x - 1; ++x) {
            temp_data(t + 1, x) = temp_data(t, x) + r *
                (temp_data(t, x + 1) - 2 * temp_data(t, x) + temp_data(t, x - 1));
        }
        temp_data(t + 1, 0) = initial_conditions[0];
        temp_data(t + 1, n_x - 1) = initial_conditions[n_x - 1];
    }    
};

void Solver_Сrank_Nicolson::solve(Array2D<double> &temp_data, std::vector<double> initial_conditions, 
    const double a, const int n_x, const int n_t, const double length, const double t_max) const
{
    double dx = length / (n_x - 1);
    double dt = t_max / (n_t - 1);
    double r = a * a * dt / (dx * dx);

    for (size_t i = 0; i < n_x; ++i) {
        temp_data(0, i) = initial_conditions[i];
    }

    std::vector<double> alpha(n_x - 1, 0.0);
    std::vector<double> beta(n_x - 1, 0.0);
    std::vector<double> gamma(n_x - 1, 0.0);
    std::vector<double> b(n_x - 1, 0.0);
    std::vector<double> u(n_x - 1, 0.0);

    for (size_t t = 0; t < n_t - 1; ++t) {
        #pragma omp parallel for
        for (size_t i = 1; i < n_x - 1; ++i) {
            alpha[i] = -r / 2;
            beta[i] = 1 + r;
            gamma[i] = -r / 2;
            b[i] = temp_data(t, i) + (r / 2) * (temp_data(t, i + 1) - 2 * temp_data(t, i) + temp_data(t, i - 1));
        }

        // Метод прогонки 
        for (size_t i = 1; i < n_x - 2; ++i) {
            double m = alpha[i + 1] / beta[i];
            beta[i + 1] -= m * gamma[i];
            b[i + 1] -= m * b[i];
        }

        u[n_x - 2] = b[n_x - 2] / beta[n_x - 2];
        for (int i = n_x - 3; i >= 1; --i) {
            u[i] = (b[i] - gamma[i] * u[i + 1]) / beta[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 1; i < n_x - 1; ++i) {
            temp_data(t + 1, i) = u[i];
        }

        // Граничные условия
        temp_data(t + 1, 0) = initial_conditions[0];
        temp_data(t + 1, n_x - 1) = initial_conditions[n_x - 1];
    }    
};


void Solver_Crank_Nicolson_modified::solve(Array2D<double> &temp_data, std::vector<double> initial_conditions, 
    const double a, const int n_x, const int n_t, 
    const double length, const double t_max, const double h, const double u_c) const
{
    double dx = length / (n_x - 1);
    double dt = t_max / (n_t - 1);
    double r = a * a * dt / (dx * dx);
    
    for (size_t i = 0; i < n_x; ++i) {
        temp_data(0, i) = initial_conditions[i];
    }

    std::vector<double> alpha(n_x - 1, 0.0);
    std::vector<double> beta(n_x - 1, 0.0);
    std::vector<double> gamma(n_x - 1, 0.0);
    std::vector<double> b(n_x - 1, 0.0);
    std::vector<double> u(n_x - 1, 0.0);

    for (size_t t = 0; t < n_t - 1; ++t) {
        #pragma omp parallel for
        for (size_t i = 1; i < n_x - 1; ++i) {
            alpha[i] = -r / 2;
            beta[i] = 1 + r + h * dt;  
            gamma[i] = -r / 2;
            b[i] = temp_data(t, i) + (r / 2) * (temp_data(t, i + 1) - 2 * temp_data(t, i) + temp_data(t, i - 1))
                   + h * dt * u_c;  
        }

        // Метод прогонки 
        for (size_t i = 1; i < n_x - 2; ++i) {
            double m = alpha[i + 1] / beta[i];
            beta[i + 1] -= m * gamma[i];
            b[i + 1] -= m * b[i];
        }

        u[n_x - 2] = b[n_x - 2] / beta[n_x - 2];
        for (int i = n_x - 3; i >= 1; --i) {
            u[i] = (b[i] - gamma[i] * u[i + 1]) / beta[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 1; i < n_x - 1; ++i) {
            temp_data(t + 1, i) = u[i];
        }

        // Граничные условия
        temp_data(t + 1, 0) = initial_conditions[0];
        temp_data(t + 1, n_x - 1) = initial_conditions[n_x - 1];
    }    
};
