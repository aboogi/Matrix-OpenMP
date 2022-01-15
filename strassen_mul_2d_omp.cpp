
// Функции реализации алгоритма Штрассена

#include <iostream>
#include <vector>

#include "omp.h"
// #include "strassen_mul_2d_omp.h"

using namespace std;

namespace str_omp_2d
{
	struct Matrix
	{
		vector<vector<long>> M11;
		vector<vector<long>> M12;
		vector<vector<long>> M21;
		vector<vector<long>> M22;
	};

	// Simple matrix mult
	void matrix_mul_2d(
		std::vector<vector<long>>& mat1,  // матрица А(n*m)
		size_t s1_n,					  // размерность n матрицы А
		std::vector<vector<long>>& mat2,  // матрица B(m*k)
		size_t s2_n,					  // размерность m матрицы B
		std::vector<vector<long>>& m_res, // матрица C(n*k)
		size_t mr_m						  // размерность k матрицы C)
	)
	{
		//cout << "matrix_mul_2d in Stassen_2d_omp, threads = " << omp_get_num_threads() << endl;
		for (long i = 0; i < s1_n; i++)
		{
			for (long j = 0; j < mr_m; j++)
			{
				long sum = 0;
				for (long k = 0; k < s2_n; k++)
				{
					sum += mat1[i][k] * mat2[k][j];
				}
				m_res[i][j] = sum;
			}
		}
	}

	// Summary of matrixs M1 + M2 = ResM
	void sum_matrix_2d(vector<vector<long>>& M1, vector<vector<long>>& M2, vector<vector<long>>& ResM, size_t n, int threads)
	{
		long i;
		// #pragma omp parallel if (n > 255) privat(i) shared(M1, M2, ResM)
		// 		{
		// 			//cout << "sum_matrix_2d: threads = " << omp_get_num_threads() << endl;
		// #pragma omp for
		for (i = 0; i < n; i++)
		{
			for (long j = 0; j < n; j++)
			{
				// #pragma omp critical
				// {
				ResM[i][j] = M1[i][j] + M2[i][j];
				// }
				// }
			}
		}
	}

	// Subtraction of matrixs M1 - M2 = ResM
	void sub_matrix_2d(vector<vector<long>>& M1, vector<vector<long>>& M2, vector<vector<long>>& ResM, size_t n, int threads)
	{
		long i;
		// #pragma omp parallel if(n > 255)  privat(i) shared(M1, M2, ResM)
		// 		{
		// 			//cout << "sub_matrix_2d: threads = " << omp_get_num_threads() << endl;
		// #pragma omp for
		for (i = 0; i < n; i++)
		{
			for (long j = 0; j < n; j++)
			{
				// #pragma omp critical
				// 					{
				ResM[i][j] = M1[i][j] - M2[i][j];
				// }
				// }
			}
		}
	}

	void corners(vector<vector<long>>& matrix, size_t n, size_t m,
		vector<vector<long>>& m11, vector<vector<long>>& m12, vector<vector<long>>& m21, vector<vector<long>>& m22, int threads)
	{
		long n_2 = n / 2;
		long m_2 = m / 2;
		long t;

		// #pragma omp parallel if(n_2 > 255) privat(t) shared(m11, m12, m21, m22, matrix)
		// 		{
		// 			//cout << "corners: threads = " << omp_get_num_threads() << endl;
		// #pragma omp for
		for (t = 0; t < n_2; t++)
		{
			for (long p = 0; p < m_2; p++)
			{
				// #pragma omp critical
				// 					{
				m11[t][p] = matrix[t][p];
				m12[t][p] = matrix[t][p + m_2];
				m21[t][p] = matrix[t + n_2][p];
				m22[t][p] = matrix[t + n_2][p + m_2];
				// }
				// }
			}
		}
	}

	void fill_corners(vector<vector<long>>& C_M11, vector<vector<long>>& C_M12,
		vector<vector<long>>& C_M21, vector<vector<long>>& C_M22,
		long n, vector<vector<long>>& res_m, int threads)
	{
		// long* corner = new long[n * m / 4];
		long n_2 = n / 2;
		long m_2 = n / 2;
		long t(0);
		// #pragma omp parallel if(n_2 > 255) privat(t) shared(res_m, C_M11, C_M12, C_21, C_22)
		// 		{
		// 			cout << "fill_corners: threads = " << omp_get_num_threads() << endl;
		// #pragma omp for
		for (t = 0; t < n_2; t++)
		{
			for (long p = 0; p < n_2; p++)
			{
				// #pragma omp critical
				// 					{
				res_m[t][p] = C_M11[t][p];
				res_m[t][p + m_2] = C_M12[t][p];
				res_m[t + n_2][p] = C_M21[t][p];
				res_m[t + n_2][p + m_2] = C_M22[t][p];
				// }
				// }
			}
		}
	}

	void compute_matrix_strassen_omp(
		vector<vector<long>>& mat1,	   // (n*m)
		size_t n,					   //
		vector<vector<long>>& mat2,	   //
		size_t m,					   //
		vector<vector<long>>& mat_res, //
		size_t k,					   //
		int threads					   // Количество нитей
	)
	{
		//cout << "compute_matrix_strassen_omp, threads = " << omp_get_num_threads() << endl;
		if (k <= 128)
		{
			matrix_mul_2d(mat1, n, mat2, m, mat_res, k);
		}
		else
		{
			size_t n_2 = n / 2;
			size_t m_2 = m / 2;
			size_t k_2 = k / 2;

			// Matrix C;

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

			vector<vector<long>> RA1(n_2, vector<long>(m_2));
			vector<vector<long>> RB2(n_2, vector<long>(m_2));

			corners(mat1, n, m, A_M11, A_M12, A_M21, A_M22, threads);
			corners(mat2, n, m, B_M11, B_M12, B_M21, B_M22, threads);

#pragma omp parallel firstprivate(RA1, RB2)
			{
#pragma omp single
				{
#pragma omp task
					{
						// S1
						sub_matrix_2d((B_M12), (B_M22), RB2, n_2, threads);
						compute_matrix_strassen_omp(A_M11, n_2, RB2, n_2, P1, k_2, threads);
					}
#pragma omp task
					{
						// S2
						sum_matrix_2d((A_M11), (A_M12), RA1, n_2, threads);
						compute_matrix_strassen_omp(RA1, n_2, B_M22, n_2, P2, k_2, threads);
					}
#pragma omp task
					{
						// S3
						sum_matrix_2d((A_M21), (A_M22), RA1, n_2, threads);
						compute_matrix_strassen_omp(RA1, n_2, B_M11, n_2, P3, k_2, threads);
					}
#pragma omp task
					{
						// S4
						sub_matrix_2d((B_M21), (B_M11), RB2, n_2, threads);
						compute_matrix_strassen_omp(A_M22, n_2, RB2, n_2, P4, k_2, threads);
					}
#pragma omp task
					{
						// S5 = A11 + A22 == RA1
						// S6 = B11 + B22 == RB2
						// P1 == P5
						sum_matrix_2d((A_M11), (A_M22), RA1, n_2, threads);
						sum_matrix_2d((B_M11), (B_M22), RB2, n_2, threads);
						compute_matrix_strassen_omp(RA1, n_2, RB2, n_2, P5, k_2, threads);
					}
#pragma omp task
					{
						// S7
						// S8
						sub_matrix_2d((A_M12), (A_M22), RA1, n_2, threads);
						sum_matrix_2d((B_M21), (B_M22), RB2, n_2, threads);
						compute_matrix_strassen_omp(RA1, n_2, RB2, n_2, P6, k_2, threads);
					}
#pragma omp task
					{
						//S9
						//S10
						sub_matrix_2d((A_M11), (A_M21), RA1, n_2, threads);
						sum_matrix_2d((B_M11), (B_M12), RB2, n_2, threads);
						compute_matrix_strassen_omp(RA1, n_2, RB2, n_2, P7, k_2, threads);
					}
				};
#pragma omp taskwait
#pragma omp single
				{

#pragma omp task
					{
						sum_matrix_2d(P5, P4, RA1, n_2, threads);
						sub_matrix_2d(RA1, P2, RB2, n_2, threads);
						sum_matrix_2d(RB2, P6, C_M11, n_2, threads);
					}
#pragma omp task
					{
						sum_matrix_2d(P3, P4, C_M21, n_2, threads);
					}
#pragma omp task
					{
						sum_matrix_2d(P1, P2, C_M12, n_2, threads);
					}
#pragma omp task
					{
						sum_matrix_2d(P5, P1, RA1, n_2, threads);
						sub_matrix_2d(RA1, P3, RB2, n_2, threads);
						sub_matrix_2d(RB2, P7, C_M22, n_2, threads);
					}
				};
			}

			fill_corners(C_M11, C_M12, C_M21, C_M22, n, mat_res, threads);
		}
	}
}

void matrix_strassen_2d_omp(
	vector<vector<long>>& mat1,
	size_t n, //
	vector<vector<long>>& mat2,
	size_t m, //
	vector<vector<long>>& mat_res,
	size_t k, //
	int threads)
{
	//cout << "TUT" << omp_get_num_threads() << endl;
	str_omp_2d::compute_matrix_strassen_omp(mat1, m, mat2, n, mat_res, k, threads);
}