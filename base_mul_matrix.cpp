#include <cstdio>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

void matrix_mul_1d_omp(
    std::vector<long> *mat1,
    size_t s1_n,
    std::vector<long> *mat2,
    size_t s2_n,
    std::vector<long> *m_res,
    size_t mr_m,
    int threads)
{
    long sum;

    long i, j, k;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

#pragma omp parallel private(i, j, k, sum) shared(mat1, mat2, m_res)
    {
        //#pragma omp for schedule (static)
        printf("(matrix_mul_2d_omp) num_threads = %d\n", omp_get_num_threads());
#pragma omp for 
        for (i = 0; i < s1_n; i++)
        {
            for (j = 0; j < mr_m; j++)
            {
                sum = 0;
                for (k = 0; k < s2_n; k++)
                {
                    sum += (*mat1)[i * s2_n + k] * (*mat2)[k * mr_m + j];
                }
#pragma omp critical
                {
                    (*m_res)[i * mr_m + j] = sum;
                }
            }
        }
    }
}

void matrix_mul_1d(
    std::vector<long> *mat1,  // матрица А(n*m)
    size_t s1_n,              // размерность n матрицы А
    std::vector<long> *mat2,  // матрица B(m*k)
    size_t s2_n,              // размерность m матрицы B
    std::vector<long> *m_res, // матрица C(n*k)
    size_t mr_m               // размерность k матрицы C)
)
{
    long sum;
    long i, j, k;

    for (i = 0; i < s1_n; i++)
    {
        for (j = 0; j < mr_m; j++)
        {
            sum = 0;
            for (k = 0; k < s2_n; k++)
            {
                sum += (*mat1)[i * s2_n + k] * (*mat2)[k * mr_m + j];
            }
            (*m_res)[i * mr_m + j] = sum;
        }
    }
}

void matrix_mul_2d(
    std::vector<std::vector<long>> *mat1,  // матрица А(n*m)
    size_t s1_n,                           // размерность n матрицы А
    std::vector<std::vector<long>> *mat2,  // матрица B(m*k)
    size_t s2_n,                           // размерность m матрицы B
    std::vector<std::vector<long>> *m_res, // матрица C(n*k)
    size_t mr_m                            // размерность k матрицы C)
)
{
    long sum;
    long i, j, k;

    for (i = 0; i < s1_n; i++)
    {
        for (j = 0; j < mr_m; j++)
        {
            sum = 0;
            for (k = 0; k < s2_n; k++)
            {
                sum += (*mat1)[i][k] * (*mat2)[k][j];
            }
            (*m_res)[i][j] = sum;
        }
    }
}

void matrix_mul_2d_omp(
    std::vector<std::vector<long>> *mat1,
    size_t s1_n,
    std::vector<std::vector<long>> *mat2,
    size_t s2_n,
    std::vector<std::vector<long>> *m_res,
    size_t mr_m,
    int threads)
{
    long i(0);
    omp_set_num_threads(threads);

#pragma omp parallel private(i) shared(mat1, mat2, m_res)
    {
        //printf("(matrix_mul_2d_omp) num_threads = %d\n", omp_get_num_threads());
#pragma omp for
        for (i = 0; i < s1_n; i++)
        {
            for (long j = 0; j < mr_m; j++)
            {
                long sum = 0;
                for (long k = 0; k < s2_n; k++)
                {
                    sum += (*mat1)[i][k] * (*mat2)[k][j];
                }
#pragma omp critical
                {
                    (*m_res)[i][j] = sum;
                }
            }
        }
    }
}
