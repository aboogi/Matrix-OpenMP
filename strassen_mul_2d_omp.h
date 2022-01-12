#pragma once
#include <vector>


void matrix_strassen_2d_omp(
	std::vector<std::vector<long>>* mat1, // ������� �(n*m)
	size_t n, // ����������� n ������� �
	std::vector<std::vector<long>>* mat2, // ������� B(m*k)
	size_t m, // ����������� m ������� B
	std::vector<std::vector<long>>* mat_res,// ������� C(n*k)
	size_t k, // ����������� k ������� C)
	int threads
);
