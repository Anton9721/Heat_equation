import numpy as np
cimport numpy as cnp
cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.parallel import prange
from libc.math cimport sin, exp


cdef extern from "include/Library.hpp":
    vector[double] data_temp(string solver_name, 
                                vector[double] initial_conditions, 
                                double a, 
                                int n_x, 
                                int n_t, 
                                double length,
                                double t_max,
                                double h, 
                                double u_c)
    

cdef vector[double] python_list_to_vector(cnp.ndarray[cnp.double_t, ndim=1] np_array):
    cdef vector[double] cpp_vector
    for i in range(np_array.shape[0]):
        cpp_vector.push_back(np_array[i])
    return cpp_vector

def vector_to_numpy_2d(vector[double] cpp_vector, int n_t, int n_x):
    return np.array(cpp_vector, dtype=np.float64).reshape((n_t, n_x))

def vector_to_python_list(vector[double] cpp_vector):
    return [cpp_vector[i] for i in range(cpp_vector.size())]

# Численное рещение
def temp_result(solver_name: str, initial_conditions, 
                     a: float, n_x: int, n_t: int, 
                     length: float, t_max: float, h: float = 0, u_c: float = 0):
    all_solvers = ['Euler_method', "Nicholson_Crunk_method", "Nicholson_Crunk_method_modified"]
    
    if solver_name not in all_solvers:
        print(f"Не существует метода решения {solver_name}. Проверьте правильность написания!")
        print(f"Допустимые решатели: {all_solvers}")
        return None    

    if isinstance(initial_conditions, list):
        initial_conditions = np.array(initial_conditions, dtype=np.float64)

    cdef string cpp_solver_name = solver_name.encode('utf-8')
    cdef vector[double] cpp_initial_conditions = python_list_to_vector(initial_conditions)

    if solver_name == 'Euler_method':
        dx = length / (n_x - 1)
        dt = t_max / (n_t - 1)
        r = (a * a * dt) / (dx * dx)
        if r >= 0.5:
            print(f"Метод неустойчив: r = {r}")
            return 0
    
    cdef vector[double] result_vector = data_temp(cpp_solver_name, 
                                                       cpp_initial_conditions, 
                                                       a, 
                                                       n_x, 
                                                       n_t, 
                                                       length, 
                                                       t_max,
                                                       h,
                                                       u_c)

    return vector_to_numpy_2d(result_vector, n_t, n_x)


# Аналитическое решение
cdef double Temp_analit_function(double x, double t, double a, double u_0, double length, int terms) nogil:
    cdef:
        double value = 0.0
        int k, n
        double coef, term
    
    for k in range(terms):
        n = 2 * k + 1
        coef = n * 3.141592653589793 / length  # np.pi заменено на константу
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