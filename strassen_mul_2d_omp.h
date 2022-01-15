#pragma once
#include <vector>


void matrix_strassen_2d_omp(
	std::vector<std::vector<long>>& mat1, // 
	size_t n, // 
	std::vector<std::vector<long>>& mat2, // 
	size_t m, // 
	std::vector<std::vector<long>>& mat_res,// 
	size_t k, // 
	int threads
);
