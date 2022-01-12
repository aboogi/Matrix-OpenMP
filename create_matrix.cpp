#include <chrono>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib> // ��� ������������� ������� exit()
#include <vector>
#include <random>

using namespace std;

void create_matrix()
{
    // std::random_device rd;
    // std::mt19937 mersenne(rd());

    bool random = true;
    // bool random = false;

    size_t m, n, k;
    vector<int> mat1, mat2;

    m = 8192;
    n = m;
    k = m;

    mat1 = vector<int>(m * n);
    mat2 = vector<int>(n * k);

    for (size_t i = 0; i < m * n; ++i)
    {
        mat1[i] = rand();
    }
    for (size_t i = 0; i < n * k; ++i)
    {
        mat2[i] = rand();
    }

    FILE *outf_1; // ofstream outf("res_time_compute.txt");
    FILE *outf_2;
    // char name_1[100];
    // char name_2[100];
    string name_1 = ".\\matrix\\matrix_1_[" + to_string(m) + "x" + to_string(m) + "]" + ".txt";
    string name_2 = ".\\matrix\\matrix_2_[" + to_string(m) + "x" + to_string(m) + "]" + ".txt";
    outf_1 = fopen(name_1.c_str(), "w");
    outf_2 = fopen(name_2.c_str(), "w");

    for (size_t j = 0; j < m * n; j++)
    {
        fprintf(outf_1, "%d\n", mat1[j]);
        fprintf(outf_2, "%d\n", mat2[j]);
    }
    fclose(outf_1);
    fclose(outf_2);
}

int main()
{
    create_matrix();

    return 0;
}