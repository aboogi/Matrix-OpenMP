
// Функции реализации алгоритма Штрассена

#include <vector>
#include <iostream>
#include "strassen_mul_1d.h"


using namespace std;

namespace str_base
{
    struct Matrix {
        vector<long> M11;
        vector<long> M12;
        vector<long> M21;
        vector<long> M22;
    };

    void matrix_mul_1d(
        std::vector<long>* mat1,  // матрица А(n*m)
        size_t s1_n, // размерность n матрицы А
        std::vector<long>* mat2, // матрица B(m*k)
        size_t s2_n, // размерность m матрицы B
        std::vector<long>* m_res, // матрица C(n*k)
        size_t mr_m // размерность k матрицы C)
    )
    {
        long sum;
        long i, j, k;

        for (i = 0; i < s1_n; i++) {
            for (j = 0; j < mr_m; j++) {
                sum = 0;
                for (k = 0; k < s2_n; k++) {
                    sum += (*mat1)[i * s2_n + k] * (*mat2)[k * mr_m + j];
                }
                (*m_res)[i * mr_m + j] = sum;
            }
        }
    }


    // Summary of matrixs M1 + M2 = ResM
    vector<long> sum_matrix(vector<long>* M1, vector<long>* M2, size_t n) {
        vector<long> ResM(n * n);
        for (size_t i = 0; i < n * n; i++) {
            ResM[i] = (*M1)[i] + (*M2)[i];
        }
        return ResM;
    }

    // Subtraction of matrixs M1 - M2 = ResM
    vector<long> sub_matrix(vector<long>* M1, vector<long>* M2, size_t n) {
        vector<long> ResM(n * n);
        for (size_t i = 0; i < n * n; i++) {
            ResM[i] = (*M1)[i] - (*M2)[i];
        }
        return ResM;
    }

    void corners(vector<long>* matrix, size_t n, size_t m,
        vector<long>* m11, vector<long>* m12, vector<long>* m21, vector<long>* m22)
    {
        long token = 0;
        for (long t = 0; t < n / 2; t++) {
            for (long p = 0; p < m / 2; p++) {
                //int indx_1 = t + p;
                //int indx_2 = t * m + p;
                token = ((t * n) / 2) + p;
                //cout << indx_1 << ' ' << indx_2 << endl;
                (*m11)[token] = (*matrix)[t * m + p];
                (*m12)[token] = (*matrix)[t * m + p + (m / 2)];
                (*m21)[token] = (*matrix)[((2 * t + m) * n + 2 * p) / 2];
                (*m22)[token] = (*matrix)[((2 * t + m) * n + 2 * p) / 2 + (m / 2)]; // t* m + p + (m / 2) + 2 * n];
            }
        }
    }

    void fill_corners(vector<long>* C_M11, vector<long>* C_M12,
        vector<long>* C_M21, vector<long>* C_M22,
        long n, vector<long>* res_m) {
        // long* corner = new long[n * m / 4];
        long token = 0;
        long n_2 = n / 2;

        for (long t = 0; t < n_2; t++) {
            for (long p = 0; p < n_2; p++) {
                long i_1 = t * n + p;
                token = ((t * n) / 2) + p;

                (*res_m)[t * n + p] = (*C_M11)[token];
                (*res_m)[t * n + p + n_2] = (*C_M12)[token];
                (*res_m)[((2 * t + n) * n + 2 * p) / 2] = (*C_M21)[token];
                (*res_m)[((2 * t + n) * n + 2 * p) / 2 + n_2] = (*C_M22)[token];
            }
        }
    }

    void compute_matrix_strassen_1d(
        vector<long>* mat1, // ������� �(n*m)
        size_t n, // ����������� n ������� �
        vector<long>* mat2, // ������� B(m*k)
        size_t m, // ����������� m ������� B
        vector<long>* mat_res,// ������� C(n*k)
        size_t k // ����������� k ������� C)
    )
    {
        if (k <= 256) {
            matrix_mul_1d(mat1, n, mat2, m, mat_res, k);
        }
        else {
            size_t n_2 = n / 2;
            size_t m_2 = m / 2;
            size_t k_2 = k / 2;

            Matrix A, B, C;
            // Matrix B = Matrix();
            // Matrix C = Matrix();

            A.M11 = vector<long>(n_2 * m_2), A.M12 = vector<long>(n_2 * m_2), A.M21 = vector<long>(n_2 * m_2), A.M22 = vector<long>(n_2 * m_2);
            B.M11 = vector<long>(n_2 * m_2), B.M12 = vector<long>(n_2 * m_2), B.M21 = vector<long>(n_2 * m_2), B.M22 = vector<long>(n_2 * m_2);
            C.M11 = vector<long>(n_2 * m_2), C.M12 = vector<long>(n_2 * m_2), C.M21 = vector<long>(n_2 * m_2), C.M22 = vector<long>(n_2 * m_2);

            corners(mat1, n, m, &A.M11, &A.M12, &A.M12, &A.M12);
            corners(mat1, n, m, &B.M11, &B.M12, &B.M12, &B.M12);

            //left_up_corner(mat1, n, m, &(A.M11));
            //right_up_corner(mat1, n, m, &(A.M12));
            //left_down_corner(mat1, n, m, &(A.M21));
            //right_down_corner(mat1, n, m, &(A.M22));

            //left_up_corner(mat2, m, k, &(B.M11));
            //right_up_corner(mat2, m, k, &(B.M12));
            //left_down_corner(mat2, m, k, &(B.M21));
            //right_down_corner(mat2, m, k, &(B.M22));

            vector<long> S1 = sub_matrix(&(B.M12), &(B.M22), n_2);
            vector<long> S2 = sum_matrix(&(A.M11), &(A.M12), n_2);
            vector<long> S3 = sum_matrix(&(A.M21), &(A.M22), n_2);
            vector<long> S4 = sub_matrix(&(B.M21), &(B.M11), n_2);
            vector<long> S5 = sum_matrix(&(A.M11), &(A.M22), n_2);
            vector<long> S6 = sum_matrix(&(B.M11), &(B.M22), n_2);
            vector<long> S7 = sub_matrix(&(A.M12), &(A.M22), n_2);
            vector<long> S8 = sum_matrix(&(B.M21), &(B.M22), n_2);
            vector<long> S9 = sub_matrix(&(A.M11), &(A.M21), n_2);
            vector<long> S10 = sum_matrix(&(B.M11), &(B.M12), n_2);

            vector<long> P1(n_2 * m_2);
            vector<long> P2(n_2 * m_2);
            vector<long> P3(n_2 * m_2);
            vector<long> P4(n_2 * m_2);
            vector<long> P5(n_2 * m_2);
            vector<long> P6(n_2 * m_2);
            vector<long> P7(n_2 * m_2);

            compute_matrix_strassen_1d(&A.M11, n_2, &S1, n_2, &P1, k_2);
            compute_matrix_strassen_1d(&S2, n_2, &B.M22, n_2, &P2, k_2);
            compute_matrix_strassen_1d(&S3, n_2, &B.M11, n_2, &P3, k_2);
            compute_matrix_strassen_1d(&A.M22, n_2, &S4, n_2, &P4, k_2);
            compute_matrix_strassen_1d(&S5, n_2, &S6, n_2, &P5, k_2);
            compute_matrix_strassen_1d(&S7, n_2, &S8, n_2, &P6, k_2);
            compute_matrix_strassen_1d(&S9, n_2, &S10, n_2, &P7, k_2);

            vector<long> sum1 = sum_matrix(&P5, &P4, n_2);
            vector<long> sub1 = sub_matrix(&P6, &P2, n_2);

            vector<long> sum2 = sum_matrix(&P5, &P1, n_2);
            vector<long> sum3 = sum_matrix(&P3, &P7, n_2);

          /*  C.M11 = sum_matrix(&sum1, &sub1, n_2);
            C.M12 = sum_matrix(&P1, &P2, n_2);
            C.M21 = sum_matrix(&P3, &P4, n_2);
            C.M22 = sub_matrix(&sum2, &sum3, n_2);*/

            fill_corners(&C.M11, &C.M12, &C.M21, &C.M22, n, mat_res);
        }
    }
}

void matrix_strassen_1d(std::vector<long>* mat1, size_t n, std::vector<long>* mat2, size_t m, std::vector<long>* mat_res, size_t k)
{
    str_base::compute_matrix_strassen_1d(mat1, n, mat2, m, mat_res, k);
}
