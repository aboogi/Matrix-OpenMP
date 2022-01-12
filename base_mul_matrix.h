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
    std::vector<long>* mat1,  // ������� �(n*m)
    size_t s1_n, // ����������� n ������� �
    std::vector<long>* mat2, // ������� B(m*k)
    size_t s2_n, // ����������� m ������� B
    std::vector<long>* m_res, // ������� C(n*k)
    size_t mr_m// ����������� k ������� C
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
    std::vector<std::vector<long>>* mat1,  // ������� �(n*m)
    size_t s1_n, // ����������� n ������� �
    std::vector<std::vector<long>>* mat2, // ������� B(m*k)
    size_t s2_n, // ����������� m ������� B
    std::vector<std::vector<long>>* m_res, // ������� C(n*k)
    size_t mr_m// ����������� k ������� C
);
