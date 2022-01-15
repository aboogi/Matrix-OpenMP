#define _CRT_SECURE_NO_DEPRECATE

#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#include "base_mul_matrix.h"
#include "strassen_mul_1d.h"
#include "strassen_mul_1d_omp.h"
#include "strassen_mul_2d.h"
#include "strassen_mul_2d_omp.h"

using namespace std;

void load_matrix_1d(vector<long>* matrix1, vector<long>* matrix2, size_t m);
void load_matrix_2d(vector<vector<long>>* matrix1, vector<vector<long>>* matrix2, size_t m);
void compute_1d();
void compute_2d();

int main(int argc, char* argv[])
{
    // compute_1d();
    compute_2d();

    system("pause");
    return 0;
}

void compute_2d()
{
    vector<vector<long>> mres1;

    bool random = true;
    size_t m, n, k;
    // int th;
    vector<vector<long>> mat1, mat2;
    vector<size_t> nm = { 512, 1024, 2048 };
    vector<int> ths = { 1, 2, 3, 4, 5, 6, 7, 8 };


    for (int j = 0; j < nm.size(); j++) {

        string name = "res_time_compute-v2_2d_[" + to_string(nm[j]) + "].txt";

        std::ofstream fout(name);
        if (!fout.is_open()) {
            cerr << "Uh oh, " << name << " could not be opened for writing!" << endl;
            exit(1);
        }

        fout << "Size matrixs: [" << nm[j] << " x " << nm[j] << "]\nCount of elements: " << nm[j] * nm[j] << "\n\n";
   
        for (int i = 0; i < ths.size(); i++) {
            int th = ths[i]; // Counts of threads
            if (random) {

                m = nm[j];
                n = m;
                k = m;
               

                mat1 = vector<vector<long>>(m, vector<long>(n));
                mat2 = vector<vector<long>>(n, vector<long>(k));

                load_matrix_2d(&mat1, &mat2, m);
            } else {
                m = 4;
                n = 4;
                k = 4;
                mat1 = {
                    { 1, 2, 3, 4 },
                    { 5, 6, 7, 8 },
                    { 1, 2, 3, 4 },
                    { 5, 6, 7, 8 }
                };

                mat2 = {
                    { 1, 2, 3, 4 },
                    { 5, 6, 7, 8 },
                    { 1, 2, 3, 4 },
                    { 5, 6, 7, 8 }
                };
            }

            omp_set_dynamic(0);
            omp_set_num_threads(th);

            
            vector<vector<long>> mres2(m, vector<long>(k));
            vector<vector<long>> mres3(m, vector<long>(k));
            vector<vector<long>> mres4(m, vector<long>(k));

            // ofstream outf("res_time_compute.txt");
            // char name[100];

            fout <<  "Count of threads: " << th << "\n";
            std::cout << "Count of threads: " << th << "\n";
            std::cout << "[n x m]: [" << n << " x " << m << " ]" << endl;

            if (th == 1) {

                mres1 = vector<vector<long>>(m, vector<long>(k));
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                matrix_mul_2d(&mat1, m, &mat2, n, &mres1, k);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

                fout << "Simple mult - Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "\n";
                std::printf("Simple mul - OK\n");
            }

            // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            // matrix_mul_2d_omp(&mat1, m, &mat2, n, &mres2, k, th);
            // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            // fout <<  "Simple mult OMP - Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "\n";
            // std::printf("Simple mul OMP - OK\n");


            // Matrix Strassen 2d
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            matrix_strassen_2d(&mat1, m, &mat2, n, &mres3, k);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            fout <<  "Matrix Strassen - Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "\n";
            std::printf("Simple mul OMP - OK\n");


            // Matrix Strassen 2d
            begin = std::chrono::steady_clock::now();
            matrix_strassen_2d_omp(mat1, m, mat2, n, mres4, k, th);
            end = std::chrono::steady_clock::now();

            fout <<  "Matrix Strassen OMP - Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "\n";
            std::printf("Strassen matrix mult OMP - OK\n");

            fout << std::endl;

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    // cout << mres1[i*mr_m + j] << " ";
                    if (mres1[i][j] != mres3[i][j]) {
                        cout << mres1[i][j] << " " << mres3[i][j] << i << " " << j << endl;
                        cerr << "Error?";
                        break;
                    }
                }
                //cout << endl;
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    // cout << mres1[i*mr_m + j] << " ";
                    if (mres3[i][j] != mres4[i][j]) {
                        cout << mres3[i][j] << " " << mres4[i][j] << i << " " << j << endl;
                        cerr << "Error? ";
                        break;
                    }
                }
                //cout << endl;
            }
            std::cout << "OK" << endl;
        }
    }
}

void compute_1d()
{
    bool random = true;
    // bool random = false;

    size_t m, n, k;
    // int th;
    vector<long> mat1, mat2;
    vector<size_t> nm = { 512, 1024, 2048 };
    vector<int> ths = { 1, 2, 3, 4, 5, 6, 7, 8 };
    for (int i = 0; i < ths.size(); i++) {
        for (int j = 0; j < nm.size(); j++) {
            int th = ths[i]; // Counts of threads
            if (random) {

                m = nm[j];
                n = m;
                k = m;

                mat1 = vector<long>(m * n);
                mat2 = vector<long>(n * k);

                load_matrix_1d(&mat1, &mat2, m);

                // for (size_t i = 0; i < m * n; ++i) {
                //     mat1[i] = rand();
                // }
                // for (size_t i = 0; i < n * k; ++i) {
                //     mat2[i] = rand();
                // }
            } else {
                m = 4;
                n = 4;
                k = 4;
                mat1 = { 1, 2, 3, 4,
                    5, 6, 7, 8,
                    1, 2, 3, 4,
                    5, 6, 7, 8 };

                mat2 = { 1, 2, 3, 4,
                    5, 6, 7, 8,
                    1, 2, 3, 4,
                    5, 6, 7, 8 };
            }

            omp_set_dynamic(0);
            omp_set_num_threads(th);

            vector<long> mres1(m * k);
            vector<long> mres2(m * k);
            vector<long> mres3(m * k);
            vector<long> mres4(m * k);

            time_t begin, end;

            FILE* outf; // ofstream outf("res_time_compute.txt");
            // char name[100];
            string name = "res_time_compute_[" + to_string(th) + "].txt";
            outf = fopen(name.c_str(), "a+");

            fprintf(outf, "Count of threads: %i\n", th);
            cout << "Count of threads: " << th << "\n";
            cout << "[n x m]: [" << n << " x " << m << " ]" << endl;

            fprintf(outf, "Size matrixs: [%ld x %ld]\nCount of elements: %d\n", m, k, m * k);

            if (!outf) {
                cerr << "Uh oh, res_time_compute.txt could not be opened for writing!" << endl;
                exit(1);
            }

            if (th == 1 && m < 2049) {
                begin = clock();
                matrix_mul_1d(&mat1, m, &mat2, n, &mres1, k);
                end = clock();

                fprintf(outf, "Simple mult - Time: %d\n", (end - begin));
                printf("Simple matrix mul - OK\n");

                begin = clock();
                matrix_strassen_1d(&mat1, m, &mat2, n, &mres3, k);
                end = clock();

                fprintf(outf, "Strassen matrix mult - Time: %d\n", (end - begin));
                printf("Strassen matrix mult - OK\n");
            }

            begin = clock();
            //			matrix_mul_1d_omp(&mat1, m, &mat2, n, &mres2, k, th);
            end = clock();

            fprintf(outf, "OpenMP simple mult - Time: %d\n", (end - begin));
            printf("Simple matrix mul OMP - OK\n");

            begin = clock();
            matrix_strassen_1d(&mat1, m, &mat2, n, &mres3, k);
            end = clock();

            fprintf(outf, "Strassen matrix mult - Time: %d\n", (end - begin));
            printf("Strassen matrix mult - OK\n");

            begin = clock();
            matrix_strassen_1d_omp(&mat1, m, &mat2, n, &mres4, k, th);
            end = clock();

            fprintf(outf, "Strassen matrix mult OMP- Time: %d\n", (end - begin));
            printf("Strassen matrix mult OMP - OK\n");

            fprintf(outf, "\n");
            fclose(outf);

            // for (size_t i = 0; i < n; ++i) {
            //	for (size_t j = 0; j < m; ++j) {
            //		//cout << mres1[i*mr_m + j] << " ";
            //		if (mres1[i * m + j] >= mres2[i * m + j] * 1.1 || mres1[i * m + j] <= mres2[i * m + j] * 0.9) {
            //			cerr << "Error?";
            //			break;
            //		}
            //	}
            //	//cout << endl;
            // }
            cout << "OK" << endl;
        }
    }
}

void load_matrix_1d(vector<long>* matrix1, vector<long>* matrix2, size_t m)
{
    // FILE* outf_1, *outf_2; //ofstream outf("res_time_compute.txt");
    string name_1, name_2;
    switch (m) {
    case 512:
        name_1 = ".\\matrix\\matrix_1_[512x512].txt";
        name_2 = ".\\matrix\\matrix_2_[512x512].txt";
        break;
    case 1024:
        name_1 = ".\\matrix\\matrix_1_[1024x1024].txt";
        name_2 = ".\\matrix\\matrix_2_[1024x1024].txt";
        break;
    case 2048:
        name_1 = ".\\matrix\\matrix_1_[2048x2048].txt";
        name_2 = ".\\matrix\\matrix_2_[2048x2048].txt";
        break;
    case 4096:
        name_1 = ".\\matrix\\matrix_1_[2048x2048].txt";
        name_2 = ".\\matrix\\matrix_2_[2048x2048].txt";
        break;

    default:
        break;
    }
    // outf_1 = fopen(name_1.c_str(), "r");
    // outf_2 = fopen(name_2.c_str(), "r");

    ifstream f_mat1(name_1);
    ifstream f_mat2(name_2);

    string a, b;
    long token = 0;

    while (getline(f_mat1, a)) {
        long f = stof(a);
        (*matrix1)[token] = f;
        token++;
    }

    token = 0;
    while (getline(f_mat2, b)) {
        long f = stof(b);
        (*matrix2)[token] = f;
        token++;
    }
}

void load_matrix_2d(vector<vector<long>>* matrix1, vector<vector<long>>* matrix2, size_t m)
{
    // FILE* outf_1, *outf_2; //ofstream outf("res_time_compute.txt");
    string name_1, name_2;
    switch (m) {
    case 512:
        name_1 = ".\\matrix\\matrix_1_[512x512].txt";
        name_2 = ".\\matrix\\matrix_2_[512x512].txt";
        break;
    case 1024:
        name_1 = ".\\matrix\\matrix_1_[1024x1024].txt";
        name_2 = ".\\matrix\\matrix_2_[1024x1024].txt";
        break;
    case 2048:
        name_1 = ".\\matrix\\matrix_1_[2048x2048].txt";
        name_2 = ".\\matrix\\matrix_2_[2048x2048].txt";
        break;

    default:
        break;
    }
    // outf_1 = fopen(name_1.c_str(), "r");
    // outf_2 = fopen(name_2.c_str(), "r");

    ifstream f_mat1(name_1);
    ifstream f_mat2(name_2);

    string a, b;

    for (long i = 0; i < m; i++) {
        for (long j = 0; j < m && getline(f_mat1, a); j++) {
            long f = stof(a);
            (*matrix1)[i][j] = f;
        }
    }

    for (long i = 0; i < m; i++) {
        for (long j = 0; j < m && getline(f_mat2, b); j++) {
            long f = stof(b);
            (*matrix2)[i][j] = f;
        }
    }
}
