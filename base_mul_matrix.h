#pragma once
#include <vector>


void matrix_mul_1d_omp(
    std::vector<long>* mat1,
    size_t s1_n,
    std::vector<long>* mat2,
    size_t s2_n,
    std::vector<long>* m_res,
    size_t mr_m,
    int threads
);

void matrix_mul_1d(
    std::vector<long>* mat1,  // матрица ј(n*m)
    size_t s1_n, // размерность n матрицы ј
    std::vector<long>* mat2, // матрица B(m*k)
    size_t s2_n, // размерность m матрицы B
    std::vector<long>* m_res, // матрица C(n*k)
    size_t mr_m// размерность k матрицы C
);

void matrix_mul_2d_omp(
    std::vector<std::vector<long>>* mat1,
    size_t s1_n,
    std::vector<std::vector<long>>* mat2,
    size_t s2_n,
    std::vector<std::vector<long>>* m_res,
    size_t mr_m,
    int threads
);

void matrix_mul_2d(
    std::vector<std::vector<long>>* mat1,  // матрица ј(n*m)
    size_t s1_n, // размерность n матрицы ј
    std::vector<std::vector<long>>* mat2, // матрица B(m*k)
    size_t s2_n, // размерность m матрицы B
    std::vector<std::vector<long>>* m_res, // матрица C(n*k)
    size_t mr_m// размерность k матрицы C
);
