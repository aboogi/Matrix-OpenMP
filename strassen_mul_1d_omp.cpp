
// Функции реализации алгоритма Штрассена

#include <iostream>
#include <vector>

#include "omp.h"

using namespace std;

namespace str_omp {
struct Matrix {
    vector<long> M11;
    vector<long> M12;
    vector<long> M21;
    vector<long> M22;
};
void matrix_mul_1d(
    std::vector<long>* mat1, // матрица А(n*m)
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
void sum_matrix(vector<long>* M1, vector<long>* M2, vector<long>* ResM, size_t n, int threads)
{

    //omp_set_dynamic(0);
    //omp_set_num_threads(threads);
    //cout << omp_get_num_threads() << endl;
//#pragma omp parallel for if (n > 255) schedule(static)
    for (long i = 0; i < n * n; i++) {
        //cout << "TUT sub_matrix: " << omp_get_num_threads() << endl;
        (*ResM)[i] = (*M1)[i] + (*M2)[i];
    }
}

// Subtraction of matrixs M1 - M2 = ResM
void sub_matrix(vector<long>* M1, vector<long>* M2, vector<long>* ResM, size_t n, int threads)
{

//#pragma omp parallel if (n > 255)
//    {
        cout << "TUT sub_matrix: " << omp_get_num_threads() << " " << omp_get_thread_num() << endl;
//#pragma omp for schedule(static)
        for (long i = 0; i < n * n; i++) {
            (*ResM)[i] = (*M1)[i] - (*M2)[i];
        }
//    }
}

void corners(vector<long>* matrix, size_t n, size_t m,
    vector<long>* m11, vector<long>* m12, vector<long>* m21, vector<long>* m22, int threads)
{
    long token = 0;
    cout << threads << endl;
//#pragma omp parallel if (n > 255) schedule(static)
//    {
        cout << "TUT sub_matrix: " << omp_get_num_threads() << " " << omp_get_thread_num() << endl;
//#pragma omp for private(token)
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
//    }
}

void fill_corners(vector<long>* C_M11, vector<long>* C_M12,
    vector<long>* C_M21, vector<long>* C_M22,
    long n, vector<long>* res_m, int threads)
{
    // long* corner = new long[n * m / 4];
    long token = 0;
    long n_2 = n / 2;

    //cout << omp_get_num_threads() << endl;
//#pragma omp parallel if (n_2 > 255) schedule(static)
//    {
        cout << "TUT sub_matrix: " << omp_get_num_threads() << " " << omp_get_thread_num() << endl;
//#pragma omp for private(token)
        for (long t = 0; t < n_2; t++) {
            for (long p = 0; p < n_2; p++) {
                token = ((t * n) / 2) + p;

                (*res_m)[t * n + p] = (*C_M11)[token];
                (*res_m)[t * n + p + n_2] = (*C_M12)[token];
                (*res_m)[((2 * t + n) * n + 2 * p) / 2] = (*C_M21)[token];
                (*res_m)[((2 * t + n) * n + 2 * p) / 2 + n_2] = (*C_M22)[token];
            }
        }
//    }
}

void compute_matrix_strassen_omp(
    vector<long>* mat1, // (n*m)
    size_t n, //
    vector<long>* mat2, //
    size_t m, //
    vector<long>* mat_res, //
    size_t k, //
    int threads // Количество нитей
)
{
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    if (k <= 128) {
        matrix_mul_1d(mat1, n, mat2, m, mat_res, k);
    } else {
        size_t n_2 = n / 2;
        size_t m_2 = m / 2;
        size_t k_2 = k / 2;

        Matrix C;

        vector<long> A_M11 = vector<long>(n_2 * m_2), A_M12 = vector<long>(n_2 * m_2), A_M21 = vector<long>(n_2 * m_2), A_M22 = vector<long>(n_2 * m_2);
        vector<long> B_M11 = vector<long>(n_2 * m_2), B_M12 = vector<long>(n_2 * m_2), B_M21 = vector<long>(n_2 * m_2), B_M22 = vector<long>(n_2 * m_2);
        vector<long> C_M11 = vector<long>(n_2 * m_2), C_M12 = vector<long>(n_2 * m_2), C_M21 = vector<long>(n_2 * m_2), C_M22 = vector<long>(n_2 * m_2);

        vector<long> P1(n_2 * m_2);
        vector<long> P2(n_2 * m_2);
        vector<long> P3(n_2 * m_2);
        vector<long> P4(n_2 * m_2);
        vector<long> P5(n_2 * m_2);
        vector<long> P6(n_2 * m_2);
        vector<long> P7(n_2 * m_2);

        vector<long> S1(n_2 * m_2);
        vector<long> S2(n_2 * m_2);
        vector<long> S3(n_2 * m_2);
        vector<long> S4(n_2 * m_2);
        vector<long> S5(n_2 * m_2);
        vector<long> S6(n_2 * m_2);
        vector<long> S7(n_2 * m_2);
        vector<long> S8(n_2 * m_2);
        vector<long> S9(n_2 * m_2);
        vector<long> S10(n_2 * m_2);

        vector<long> sum1(n_2 * m_2);
        vector<long> sum2(n_2 * m_2);
        vector<long> sum3(n_2 * m_2);

        vector<long> sub1(n_2 * m_2);

        //int st = clock();
        //			omp_set_nested(1);
        ////omp_set_num_threads(threads);

        corners(mat1, n, m, &A_M11, &A_M12, &A_M12, &A_M12, threads);
        corners(mat1, n, m, &B_M11, &B_M12, &B_M12, &B_M12, threads);


        sub_matrix(&(B_M12), &(B_M22), &S1, n_2, threads);
        compute_matrix_strassen_omp(&A_M11, n_2, &S1, n_2, &P1, k_2, threads);

        sum_matrix(&(A_M11), &(A_M12), &S2, n_2, threads);
        compute_matrix_strassen_omp(&S2, n_2, &B_M22, n_2, &P2, k_2, threads);

        sum_matrix(&(A_M21), &(A_M22), &S3, n_2, threads);
        compute_matrix_strassen_omp(&S3, n_2, &B_M11, n_2, &P3, k_2, threads);

        sub_matrix(&(B_M21), &(B_M11), &S4, n_2, threads);
        compute_matrix_strassen_omp(&A_M22, n_2, &S4, n_2, &P4, k_2, threads);

        sum_matrix(&(A_M11), &(A_M22), &S5, n_2, threads);
        sum_matrix(&(B_M11), &(B_M22), &S6, n_2, threads);
        compute_matrix_strassen_omp(&S5, n_2, &S6, n_2, &P5, k_2, threads);

        sub_matrix(&(A_M12), &(A_M22), &S7, n_2, threads);
        sum_matrix(&(B_M21), &(B_M22), &S8, n_2, threads);
        compute_matrix_strassen_omp(&S7, n_2, &S8, n_2, &P6, k_2, threads);

        sub_matrix(&(A_M11), &(A_M21), &S9, n_2, threads);
        sum_matrix(&(B_M11), &(B_M12), &S10, n_2, threads);
        compute_matrix_strassen_omp(&S9, n_2, &S10, n_2, &P7, k_2, threads);

        sum_matrix(&P5, &P4, &sum1, n_2, threads);
        sub_matrix(&P6, &P2, &sub1, n_2, threads);

        sum_matrix(&P5, &P1, &sum2, n_2, threads);
        sum_matrix(&P3, &P7, &sum3, n_2, threads);

        sum_matrix(&sum1, &sub1, &C_M11, n_2, threads);
        sum_matrix(&P1, &P2, &C_M12, n_2, threads);
        sum_matrix(&P3, &P4, &C_M21, n_2, threads);
        sub_matrix(&sum2, &sum3, &C_M22, n_2, threads);

        fill_corners(&C_M11, &C_M12, &C_M21, &C_M22, n, mat_res, threads);
    }
}

}

void matrix_strassen_1d_omp(
    vector<long>* mat1, // ������� �(n*m)
    size_t n, // ����������� n ������� �
    vector<long>* mat2, // ������� B(m*k)
    size_t m, // ����������� m ������� B
    vector<long>* mat_res, // ������� C(n*k)
    size_t k, // ����������� k ������� C)
    int threads)
{
    str_omp::compute_matrix_strassen_omp(mat1, m, mat2, n, mat_res, k, threads);
}
