#pragma once
#include <vector>


void matrix_strassen_1d(
    std::vector<long>* mat1,
    size_t n,
    std::vector<long>* mat2,
    size_t m,
    std::vector<long>* mat_res,
    size_t k
);
