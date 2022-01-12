
// Функции реализации алгоритма Штрассена

#include <iostream>
#include <vector>

#include "omp.h"
#include "strassen_mul_2d_omp.h"

using namespace std;

namespace str_omp_2d {
    struct Matrix {
        vector<vector<long>> M11;
        vector<vector<long>> M12;
        vector<vector<long>> M21;
        vector<vector<long>> M22;
    };

    // Simple matrix mult
    void matrix_mul_2d(
        std::vector<vector<long>>* mat1, // матрица А(n*m)
        size_t s1_n, // размерность n матрицы А
        std::vector<vector<long>>* mat2, // матрица B(m*k)
        size_t s2_n, // размерность m матрицы B
        std::vector<vector<long>>* m_res, // матрица C(n*k)
        size_t mr_m // размерность k матрицы C)
    )
    {
        long sum;
        long i, j, k;
        for (i = 0; i < s1_n; i++) {
            for (j = 0; j < mr_m; j++) {
                sum = 0;
                for (k = 0; k < s2_n; k++) {
                    sum += (*mat1)[i][k] * (*mat2)[k][j];
                }
                (*m_res)[i][j] = sum;
            }
        }
    }

    // Summary of matrixs M1 + M2 = ResM
    vector<vector<long>> sum_matrix_2d(vector<vector<long>>* M1, vector<vector<long>>* M2, size_t n, int threads)
    {
        vector<vector<long>> ResM(n, vector<long>(n));
        // omp_set_dynamic(0);
        // omp_set_num_threads(threads);
        // cout << omp_get_num_threads() << endl;
        //#pragma omp parallel if (n > 255)
        //    {
        //        cout << "sum - " << omp_get_thread_num() << endl;
        //#pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < n; j++) {
                // cout << "TUT sub_matrix: " << omp_get_num_threads() << endl;
                ResM[i][j] = (*M1)[i][j] + (*M2)[i][j];
            }
        }
        //    }
        return ResM;
    }

    // Subtraction of matrixs M1 - M2 = ResM
    vector<vector<long>> sub_matrix_2d(vector<vector<long>>* M1, vector<vector<long>>* M2, size_t n, int threads)
    {
        vector<vector<long>> ResM(n, vector<long>(n));

        //#pragma omp parallel if (n > 255)
        //    {
        //        cout << "sub - " << omp_get_thread_num() << endl;
        //#pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < n; j++) {
                ResM[i][j] = (*M1)[i][j] - (*M2)[i][j];
            }
        }
        //    }
        return ResM;
    }

    void corners(vector<vector<long>>* matrix, size_t n, size_t m,
        vector<vector<long>>* m11, vector<vector<long>>* m12, vector<vector<long>>* m21, vector<vector<long>>* m22, int threads)
    {
        long n_2 = n / 2;
        long m_2 = m / 2;

        //    cout << omp_get_num_threads() << endl;
        //#pragma omp parallel
        //    {
        //        cout << "corn - " << omp_get_thread_num() << endl;
        //#pragma omp for schedule(static)
        for (long t = 0; t < n / 2; t++) {
            for (long p = 0; p < m / 2; p++) {
                (*m11)[t][p] = (*matrix)[t][p];
                (*m12)[t][p] = (*matrix)[t][p + m_2];
                (*m21)[t][p] = (*matrix)[t + n_2][p];
                (*m22)[t][p] = (*matrix)[t + n_2][p + m_2];
            }
        }
        //    }
    }

    void fill_corners(vector<vector<long>>* C_M11, vector<vector<long>>* C_M12,
        vector<vector<long>>* C_M21, vector<vector<long>>* C_M22,
        long n, vector<vector<long>>* res_m, int threads)
    {
        // long* corner = new long[n * m / 4];
        long n_2 = n / 2;
        long m_2 = n / 2;
        //#pragma omp parallel if (n_2 > 255)
        //    {
        //        cout << "fill - " << omp_get_num_threads() << endl;
        //#pragma omp for schedule(static)
        for (long t = 0; t < n_2; t++) {
            for (long p = 0; p < n_2; p++) {
                (*res_m)[t][p] = (*C_M11)[t][p];
                (*res_m)[t][p + m_2] = (*C_M12)[t][p];
                (*res_m)[t + n_2][p] = (*C_M21)[t][p];
                (*res_m)[t + n_2][p + m_2] = (*C_M22)[t][p];
            }
        }
        //    }
    }

    void compute_matrix_strassen_omp(
        vector<vector<long>>* mat1, // (n*m)
        size_t n, //
        vector<vector<long>>* mat2, //
        size_t m, //
        vector<vector<long>>* mat_res, //
        size_t k, //
        int threads // Количество нитей
    )
    {
        omp_set_dynamic(0);
        omp_set_nested(1);
        omp_set_num_threads(threads);
#pragma omp parallel
        {
#pragma omp single
            cout << "get threads" << omp_get_num_threads() << endl;
            if (k <= 128) {
                matrix_mul_2d(mat1, n, mat2, m, mat_res, k);
            }
            else {
                size_t n_2 = n / 2;
                size_t m_2 = m / 2;
                size_t k_2 = k / 2;

                Matrix C;

                vector<vector<long>> A_M11(n_2, vector<long>(m_2)), A_M12(n_2, vector<long>(m_2)), A_M21(n_2, vector<long>(m_2)), A_M22(n_2, vector<long>(m_2));
                vector<vector<long>> B_M11(n_2, vector<long>(m_2)), B_M12(n_2, vector<long>(m_2)), B_M21(n_2, vector<long>(m_2)), B_M22(n_2, vector<long>(m_2));
                vector<vector<long>> C_M11(n_2, vector<long>(m_2)), C_M12(n_2, vector<long>(m_2)), C_M21(n_2, vector<long>(m_2)), C_M22(n_2, vector<long>(m_2));

                vector<vector<long>> P1(n_2, vector<long>(m_2));
                vector<vector<long>> P2(n_2, vector<long>(m_2));
                vector<vector<long>> P3(n_2, vector<long>(m_2));
                vector<vector<long>> P4(n_2, vector<long>(m_2));
                vector<vector<long>> P5(n_2, vector<long>(m_2));
                vector<vector<long>> P6(n_2, vector<long>(m_2));
                vector<vector<long>> P7(n_2, vector<long>(m_2));

                vector<vector<long>> S1;
                vector<vector<long>> S2;
                vector<vector<long>> S3;
                vector<vector<long>> S4;
                vector<vector<long>> S5;
                vector<vector<long>> S6;
                vector<vector<long>> S7;
                vector<vector<long>> S8;
                vector<vector<long>> S9;
                vector<vector<long>> S10;

                corners(mat1, n, m, &A_M11, &A_M12, &A_M12, &A_M12, threads);
                corners(mat1, n, m, &B_M11, &B_M12, &B_M12, &B_M12, threads);
                cout << "TUT " << omp_get_num_threads() << endl;

#pragma omp task
                {
                    S1 = sub_matrix_2d(&(B_M12), &(B_M22), n_2, threads);
                    compute_matrix_strassen_omp(&A_M11, n_2, &S1, n_2, &P1, k_2, threads);
                }
#pragma omp task
                {
                    S2 = sum_matrix_2d(&(A_M11), &(A_M12), n_2, threads);
                    compute_matrix_strassen_omp(&S2, n_2, &B_M22, n_2, &P2, k_2, threads);
                }
#pragma omp task
                {
                    S3 = sum_matrix_2d(&(A_M21), &(A_M22), n_2, threads);
                    compute_matrix_strassen_omp(&S3, n_2, &B_M11, n_2, &P3, k_2, threads);
                }
#pragma omp task
                {
                    S4 = sub_matrix_2d(&(B_M21), &(B_M11), n_2, threads);
                    compute_matrix_strassen_omp(&A_M22, n_2, &S4, n_2, &P4, k_2, threads);
                }
#pragma omp task
                {
                    S5 = sum_matrix_2d(&(A_M11), &(A_M22), n_2, threads);
                    S6 = sum_matrix_2d(&(B_M11), &(B_M22), n_2, threads);
                    compute_matrix_strassen_omp(&S5, n_2, &S6, n_2, &P5, k_2, threads);
                }
#pragma omp task
                {
                    S7 = sub_matrix_2d(&(A_M12), &(A_M22), n_2, threads);
                    S8 = sum_matrix_2d(&(B_M21), &(B_M22), n_2, threads);
                    compute_matrix_strassen_omp(&S7, n_2, &S8, n_2, &P6, k_2, threads);
                }
#pragma omp task
                {
                    S9 = sub_matrix_2d(&(A_M11), &(A_M21), n_2, threads);
                    S10 = sum_matrix_2d(&(B_M11), &(B_M12), n_2, threads);
                    compute_matrix_strassen_omp(&S9, n_2, &S10, n_2, &P7, k_2, threads);
                }
#pragma omp taskwait
                vector<vector<long>> sum1;
                vector<vector<long>> sub1;
                vector<vector<long>> sum2;
                vector<vector<long>> sum3;

#pragma omp task
                {
                    sum1 = sum_matrix_2d(&P5, &P4, n_2, threads);
                }

#pragma omp task
                {
                    sub1 = sub_matrix_2d(&P6, &P2, n_2, threads);
                }
#pragma omp task
                {
                    sum2 = sum_matrix_2d(&P5, &P1, n_2, threads);
                }
#pragma omp task
                {
                    sum3 = sum_matrix_2d(&P3, &P7, n_2, threads);
                }
#pragma omp taskwait

#pragma omp task
                {
                    C_M11 = sum_matrix_2d(&sum1, &sub1, n_2, threads);
                }
#pragma omp task
                {
                    C_M12 = sum_matrix_2d(&P1, &P2, n_2, threads);
                }
#pragma omp task
                {
                    C_M21 = sum_matrix_2d(&P3, &P4, n_2, threads);
                }
#pragma omp task
                {
                    C_M22 = sub_matrix_2d(&sum2, &sum3, n_2, threads);
                }
#pragma omp taskwait

                fill_corners(&C_M11, &C_M12, &C_M21, &C_M22, n, mat_res, threads);
            }
        }
    }
}

void matrix_strassen_2d_omp(
    vector<vector<long>>* mat1,
    size_t n, //
    vector<vector<long>>* mat2,
    size_t m, //
    vector<vector<long>>* mat_res,
    size_t k, //
    int threads)
{
    cout << "TUT" << omp_get_num_threads() << endl;
    str_omp_2d::compute_matrix_strassen_omp(mat1, m, mat2, n, mat_res, k, threads);
}
