import numpy as np
cimport numpy as cnp
cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.parallel import prange
from libc.math cimport sin, exp

    
cdef extern from "include/Library.hpp":
    void data_temp(
        int solver_num,
        double* result_data,
        const double* initial_conditions,
        double a, int n_x, int n_t,
        double length, double t_max,
        double h, double u_c)

def temp_result(int solver_num, 
               initial_conditions,
               result,
               double a, int n_x, int n_t, 
               double length, double t_max, 
               double h=0.0, double u_c=0.0):
    
    if solver_num == 0:
        dx = length / (n_x - 1)
        dt = t_max / (n_t - 1)
        r = (a * a * dt) / (dx * dx)
        if r >= 0.5:
            print(f"Метод неустойчив: r = {r}")
            return 0

    cdef cnp.double_t [:] init_view = np.ascontiguousarray(initial_conditions, dtype=np.float64)
    cdef cnp.double_t [:, :] result_view = np.ascontiguousarray(result, dtype=np.float64)    

    data_temp(
        solver_num,
        &result_view[0,0],
        &init_view[0],
        a, n_x, n_t,
        length, t_max,
        h, u_c
    )


# Аналитическое решение
cdef double Temp_analit_function(double x, double t, double a, double u_0, double length, int terms) nogil:
    cdef:
        double value = 0.0
        int k, n
        double coef, term
    
    for k in range(terms):
        n = 2 * k + 1
        coef = n * 3.141592653589793 / length  
        term = sin(coef * x) * exp(-(a * coef) ** 2 * t) / n
        value += term
    
    return (4 * u_0 / 3.141592653589793) * value

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def Temp_list_analit_parallel(int n_x, int n_t, double length, double t_max, double a, double u_0, int terms):
    cdef:
        cnp.ndarray[cnp.double_t, ndim=2] data_result = np.zeros((n_t, n_x), dtype=np.double)
        cdef double[:] x_values = np.linspace(0, length, n_x)
        cdef double[:] t_values = np.linspace(0, t_max, n_t)
        int i, j
    
    with nogil:
        for i in prange(n_t):
            for j in range(n_x):
                data_result[i, j] = Temp_analit_function(
                    x_values[j], t_values[i], a, u_0, length, terms
                )
    
    return np.asarray(data_result)