cmake_minimum_required(VERSION 3.9)

project(matrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenMP REQUIRED)

add_executable(matrix_multiply)

target_link_libraries(matrix_multiply
    PRIVATE
        OpenMP::OpenMP_CXX
)

set(matrix_init 
matrix/matrix_1_[1024x1024].txt
matrix/matrix_1_[2048x2048].txt
matrix/matrix_1_[4096x4096].txt
matrix/matrix_1_[512x512].txt
matrix/matrix_1_[8192x8192].txt
matrix/matrix_2_[1024x1024].txt
matrix/matrix_2_[2048x2048].txt
matrix/matrix_2_[4096x4096].txt
matrix/matrix_2_[512x512].txt
matrix/matrix_2_[8192x8192].txt
)

target_sources(matrix_multiply
    PRIVATE
        ${matrix_init}
    
        base_mul_matrix.cpp
        #create_matrix.cpp
        strassen_mul_1d_omp.cpp
        strassen_mul_1d.cpp
        strassen_mul_2d_omp.cpp
        strassen_mul_2d.cpp
        mul_matrix.cpp

        base_mul_matrix.h
        strassen_mul_1d_omp.h
        strassen_mul_1d.h
        strassen_mul_2d_omp.h
        strassen_mul_2d.h
)

# get_target_property(matrix_multiply_runtime matrix_multiply RUNTIME_OUTPUT_DIRECTORY)
file(COPY ${matrix_init} DESTINATION ${CMAKE_BINARY_DIR}/matrix)
