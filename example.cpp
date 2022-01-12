//#include <iostream>
//#include <omp.h>
//#include <future>
////#include "stdafx.h"
//
//class Profiler
//{
//public:
//    Profiler() {
//        start = std::chrono::system_clock::now();
//    }
//    ~Profiler() {
//        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
//        std::chrono::duration<float> difference = end - start;
//        std::cout << difference.count() * 1 << std::endl;
//    }
//private:
//    std::chrono::time_point<std::chrono::system_clock> start;
//};
//
//void copy(int* a, int* b, int ib, int jb, int n)
//{
//    int i, j, k;
//    int imax = ib + n / 2;      //�� ������ ������� ������ ��������
//    int jmax = jb + n / 2;      //���������� �� ��������
//    //#pragma omp parallel for
//
//    for (k = 0, i = ib; i < imax; i++)
//    {
//        for (j = jb; j < jmax; j++)
//        {
//            a[k++] = b[i * n + j];  //� ������� a �������� ���� ������
//        }
//    }
//}
//
//void copy_test(int* a, int** b, int ib, int jb, int n)
//{
//    int i, j, k;
//    int imax = ib + n / 2;      //�� ������ ������� ������ ��������
//    int jmax = jb + n / 2;      //���������� �� ��������
//    //#pragma omp parallel for
//
//    for (k = 0, i = ib; i < imax; i++)
//    {
//        for (j = jb; j < jmax; j++)
//        {
//            a[k++] = b[i][j];  //� ������� a �������� ���� ������
//        }
//    }
//}
//
////�������� � �������� ������� a ������� ����������� ������� b
////�������� � �������� a[ia][[ja]
////n - ����������� ������� a
//void copyback(int* a, int ia, int ja, int* b, int n)
//{
//    int i, j, k;
//    int imax = ia + n / 2;
//    int jmax = ja + n / 2;
//    //#pragma omp parallel for
//    for (k = 0, i = ia; i < imax; i++)
//    {
//        for (j = ja; j < jmax; j++)
//        {
//            a[i * n + j] = b[k++];
//        }
//    }
//}
//
//void copyback_test(int** a, int ia, int ja, int* b, int n)
//{
//    int i, j, k;
//    int imax = ia + n / 2;
//    int jmax = ja + n / 2;
//    //#pragma omp parallel for
//    for (k = 0, i = ia; i < imax; i++)
//    {
//        for (j = ja; j < jmax; j++)
//        {
//            a[i][j] = b[k++];
//        }
//    }
//}
//
////���������� ������� ����������� ������� c = a + b
//void add(int* c, int* a, int* b, int n)
//{
//    for (int i = 0; i < n * n; i++)
//        c[i] = a[i] + b[i];
//}
//
//void add_omp(int* c, int* a, int* b, int n)
//{
//    //#pragma omp parallel for
//    for (int i = 0; i < n * n; i++)
//        c[i] = a[i] + b[i];
//}
//
////�������� ������� ����������� ������� c = a - b
//void sub(int* c, int* a, int* b, int n)
//{
//    for (int i = 0; i < n * n; i++)
//        c[i] = a[i] - b[i];
//}
//
//void sub_omp(int* c, int* a, int* b, int n)
//{
//    //#pragma omp parallel for
//    for (int i = 0; i < n * n; i++)
//        c[i] = a[i] - b[i];
//}
//
////������� ��������� ��������� c = a * b
//void mul_normal(int* c, int* a, int* b, int n)
//{
//    int i, j, k;
//    for (i = 0; i < n; i++)
//        for (j = 0; j < n; j++)
//        {
//            c[i * n + j] = 0;
//            for (k = 0; k < n; k++)
//                c[i * n + j] += a[i * n + k] * b[k * n + j];
//        }
//}
//
////��������� ���������� ��������� - ��������� (����������� ��������� ���������)
////c = a * b
//void mul(int* c, int* a, int* b, int n)
//{
//    //��� ��������� ������ ����������, �������� ������������ ������� ���������
//    //��� �������� ������� ����� 2. ����� ����� ������ ����� 32
//    if (n <= 2)
//        mul_normal(c, a, b, n);         //������� ���������
//    else
//    {
//        int h = n / 2;                    //����� ����������� �������
//        int* M = new int[h * h * 29];       //�������� ������ ��� 29 ��������������� �������
//        //������������� ��� ����������, ��� ���������� ������
//
//        //0-3: �������� ������� A
//        copy(&M[0], a, 0, 0, n);                    //M[0] = A11
//        copy(&M[h * h], a, 0, h, n);                  //M[1] = A12
//        copy(&M[2 * h * h], a, h, 0, n);                //M[2] = A21
//        copy(&M[3 * h * h], a, h, h, n);                //M[3] = A22
//
//        //4-7: �������� ������� B
//        copy(&M[4 * h * h], b, 0, 0, n);                //M[4] = B11
//        copy(&M[5 * h * h], b, 0, h, n);                //M[5] = B12
//        copy(&M[6 * h * h], b, h, 0, n);                //M[6] = B21
//        copy(&M[7 * h * h], b, h, h, n);                //M[7] = B22
//
//        //8-15: S1 - S8
//        add(&M[8 * h * h], &M[2 * h * h], &M[3 * h * h], h);    //M[8] = S1 = A21 + A22
//        sub(&M[9 * h * h], &M[8 * h * h], &M[0], h);        //M[9] = S2 = S1 - A11
//        sub(&M[10 * h * h], &M[0], &M[2 * h * h], h);       //M[10] = S3 = A11 - A21
//        sub(&M[11 * h * h], &M[h * h], &M[9 * h * h], h);     //M[11] = S4 = A12 - S2
//        sub(&M[12 * h * h], &M[5 * h * h], &M[4 * h * h], h);   //M[12] = S5 = B12 - B11
//        sub(&M[13 * h * h], &M[7 * h * h], &M[12 * h * h], h);  //M[13] = S6 = B22 - S5
//        sub(&M[14 * h * h], &M[7 * h * h], &M[5 * h * h], h);   //M[14] = S7 = B22 - B12
//        sub(&M[15 * h * h], &M[13 * h * h], &M[6 * h * h], h);  //M[15] = S8 = S6 - B21
//        //16-22: P1 - P7: ����������� ����� ���� �� ��� ���������
//        mul(&M[16 * h * h], &M[9 * h * h], &M[13 * h * h], h);  //M[16] = P1 = S2 * S6
//        mul(&M[17 * h * h], &M[0], &M[4 * h * h], h);       //M[17] = P2 = A11 * B11
//        mul(&M[18 * h * h], &M[h * h], &M[6 * h * h], h);     //M[18] = P3 = A12 * B21
//        mul(&M[19 * h * h], &M[10 * h * h], &M[14 * h * h], h); //M[19] = P4 = S3 * S7
//        mul(&M[20 * h * h], &M[8 * h * h], &M[12 * h * h], h);  //M[20] = P5 = S1 * S5
//        mul(&M[21 * h * h], &M[11 * h * h], &M[7 * h * h], h);  //M[21] = P6 = S4 * B22
//        mul(&M[22 * h * h], &M[3 * h * h], &M[15 * h * h], h);  //M[22] = P7 = A22 * S8
//        //23-24: T1, T2
//        add(&M[23 * h * h], &M[16 * h * h], &M[17 * h * h], h); //M[23] = T1 = P1 + P2
//        add(&M[24 * h * h], &M[23 * h * h], &M[19 * h * h], h); //M[24] = T2 = T1 + P4
//        //25-28: �������� �������������� ������� C
//        add(&M[25 * h * h], &M[17 * h * h], &M[18 * h * h], h); //M[25] = C11 = P2 + P3
//        add(&M[26 * h * h], &M[23 * h * h], &M[20 * h * h], h); //M[26] = C12 = T1 + P5
//        add(&M[26 * h * h], &M[26 * h * h], &M[21 * h * h], h); //M[26] = C12 += P6
//        sub(&M[27 * h * h], &M[24 * h * h], &M[22 * h * h], h); //M[27] = C21 = T2 - P7
//        add(&M[28 * h * h], &M[24 * h * h], &M[20 * h * h], h); //M[28] = C22 = T2 + P5
//        //�������� ���������
//        copyback(c, 0, 0, &M[25 * h * h], n);           //C11 = M[25]
//        copyback(c, 0, h, &M[26 * h * h], n);           //C12 = M[26]
//        copyback(c, h, 0, &M[27 * h * h], n);           //C21 = M[27]
//        copyback(c, h, h, &M[28 * h * h], n);           //C22 = M[28]
//
//        delete[]M;
//    }
//}
//
//void mul_test(int** c, int** a, int** b, int n)
//{
//    //��� ��������� ������ ����������, �������� ������������ ������� ���������
//    //��� �������� ������� ����� 2. ����� ����� ������ ����� 32
//    if (n <= 2)
//        std::cout << "---";         //������� ���������
//    else
//    {
//        int h = n / 2;                    //����� ����������� �������
//        int* M = new int[h * h * 29];       //�������� ������ ��� 29 ��������������� �������
//        //������������� ��� ����������, ��� ���������� ������
//
//        //0-3: �������� ������� A
//        copy_test(&M[0], a, 0, 0, n);                    //M[0] = A11
//        copy_test(&M[h * h], a, 0, h, n);                  //M[1] = A12
//        copy_test(&M[2 * h * h], a, h, 0, n);                //M[2] = A21
//        copy_test(&M[3 * h * h], a, h, h, n);                //M[3] = A22
//
//        //4-7: �������� ������� B
//        copy_test(&M[4 * h * h], b, 0, 0, n);                //M[4] = B11
//        copy_test(&M[5 * h * h], b, 0, h, n);                //M[5] = B12
//        copy_test(&M[6 * h * h], b, h, 0, n);                //M[6] = B21
//        copy_test(&M[7 * h * h], b, h, h, n);                //M[7] = B22
//
//        //8-15: S1 - S8
//        add(&M[8 * h * h], &M[2 * h * h], &M[3 * h * h], h);    //M[8] = S1 = A21 + A22
//        sub(&M[9 * h * h], &M[8 * h * h], &M[0], h);        //M[9] = S2 = S1 - A11
//        sub(&M[10 * h * h], &M[0], &M[2 * h * h], h);       //M[10] = S3 = A11 - A21
//        sub(&M[11 * h * h], &M[h * h], &M[9 * h * h], h);     //M[11] = S4 = A12 - S2
//        sub(&M[12 * h * h], &M[5 * h * h], &M[4 * h * h], h);   //M[12] = S5 = B12 - B11
//        sub(&M[13 * h * h], &M[7 * h * h], &M[12 * h * h], h);  //M[13] = S6 = B22 - S5
//        sub(&M[14 * h * h], &M[7 * h * h], &M[5 * h * h], h);   //M[14] = S7 = B22 - B12
//        sub(&M[15 * h * h], &M[13 * h * h], &M[6 * h * h], h);  //M[15] = S8 = S6 - B21
//        //16-22: P1 - P7: ����������� ����� ���� �� ��� ���������
//        mul(&M[16 * h * h], &M[9 * h * h], &M[13 * h * h], h);  //M[16] = P1 = S2 * S6
//        mul(&M[17 * h * h], &M[0], &M[4 * h * h], h);       //M[17] = P2 = A11 * B11
//        mul(&M[18 * h * h], &M[h * h], &M[6 * h * h], h);     //M[18] = P3 = A12 * B21
//        mul(&M[19 * h * h], &M[10 * h * h], &M[14 * h * h], h); //M[19] = P4 = S3 * S7
//        mul(&M[20 * h * h], &M[8 * h * h], &M[12 * h * h], h);  //M[20] = P5 = S1 * S5
//        mul(&M[21 * h * h], &M[11 * h * h], &M[7 * h * h], h);  //M[21] = P6 = S4 * B22
//        mul(&M[22 * h * h], &M[3 * h * h], &M[15 * h * h], h);  //M[22] = P7 = A22 * S8
//        //23-24: T1, T2
//        add(&M[23 * h * h], &M[16 * h * h], &M[17 * h * h], h); //M[23] = T1 = P1 + P2
//        add(&M[24 * h * h], &M[23 * h * h], &M[19 * h * h], h); //M[24] = T2 = T1 + P4
//        //25-28: �������� �������������� ������� C
//        add(&M[25 * h * h], &M[17 * h * h], &M[18 * h * h], h); //M[25] = C11 = P2 + P3
//        add(&M[26 * h * h], &M[23 * h * h], &M[20 * h * h], h); //M[26] = C12 = T1 + P5
//        add(&M[26 * h * h], &M[26 * h * h], &M[21 * h * h], h); //M[26] = C12 += P6
//        sub(&M[27 * h * h], &M[24 * h * h], &M[22 * h * h], h); //M[27] = C21 = T2 - P7
//        add(&M[28 * h * h], &M[24 * h * h], &M[20 * h * h], h); //M[28] = C22 = T2 + P5
//        //�������� ���������
//        copyback_test(c, 0, 0, &M[25 * h * h], n);           //C11 = M[25]
//        copyback_test(c, 0, h, &M[26 * h * h], n);           //C12 = M[26]
//        copyback_test(c, h, 0, &M[27 * h * h], n);           //C21 = M[27]
//        copyback_test(c, h, h, &M[28 * h * h], n);           //C22 = M[28]
//
//        delete[]M;
//    }
//}
//
//void mul_test_optim(int** c, int** a, int** b, int n, int threads_num)
//{
//    omp_set_num_threads(threads_num);
//    //��� ��������� ������ ����������, �������� ������������ ������� ���������
//    //��� �������� ������� ����� 2. ����� ����� ������ ����� 32
//    if (n <= 2)
//        std::cout << "---";         //������� ���������
//    else
//    {
//        int h = n / 2;                    //����� ����������� �������
//        int* M = new int[h * h * 29];       //�������� ������ ��� 29 ��������������� �������
//        //������������� ��� ����������, ��� ���������� ������
//
//
//
//        //0-3: �������� ������� A
//
//        copy_test(&M[0], a, 0, 0, n);                    //M[0] = A11
//        copy_test(&M[h * h], a, 0, h, n);                  //M[1] = A12
//        copy_test(&M[2 * h * h], a, h, 0, n);                //M[2] = A21
//        copy_test(&M[3 * h * h], a, h, h, n);                //M[3] = A22
//        //4-7: �������� ������� B
//        copy_test(&M[4 * h * h], b, 0, 0, n);                //M[4] = B11
//        copy_test(&M[5 * h * h], b, 0, h, n);                //M[5] = B12
//        copy_test(&M[6 * h * h], b, h, 0, n);                //M[6] = B21
//        copy_test(&M[7 * h * h], b, h, h, n);                //M[7] = B22
//
//#pragma omp parallel
//        {
//#pragma omp task
//            {
//                add(&M[8 * h * h], &M[2 * h * h], &M[3 * h * h], h);    //M[8] = S1 = A21 + A22
//                sub(&M[9 * h * h], &M[8 * h * h], &M[0], h);        //M[9] = S2 = S1 - A11
//                sub(&M[10 * h * h], &M[0], &M[2 * h * h], h);       //M[10] = S3 = A11 - A21
//                sub(&M[11 * h * h], &M[h * h], &M[9 * h * h], h);     //M[11] = S4 = A12 - S2
//                sub(&M[12 * h * h], &M[5 * h * h], &M[4 * h * h], h);   //M[12] = S5 = B12 - B11
//                sub(&M[13 * h * h], &M[7 * h * h], &M[12 * h * h], h);  //M[13] = S6 = B22 - S5
//                sub(&M[14 * h * h], &M[7 * h * h], &M[5 * h * h], h);   //M[14] = S7 = B22 - B12
//                sub(&M[15 * h * h], &M[13 * h * h], &M[6 * h * h], h);  //M[15] = S8 = S6 - B21
//            }
//#pragma omp taskwait
//            //16-22: P1 - P7: ����������� ����� ���� �� ��� ���������
//#pragma omp task
//            {
//                mul(&M[16 * h * h], &M[9 * h * h], &M[13 * h * h], h);  //M[16] = P1 = S2 * S6
//                mul(&M[17 * h * h], &M[0], &M[4 * h * h], h);       //M[17] = P2 = A11 * B11
//                mul(&M[18 * h * h], &M[h * h], &M[6 * h * h], h);     //M[18] = P3 = A12 * B21
//                mul(&M[19 * h * h], &M[10 * h * h], &M[14 * h * h], h); //M[19] = P4 = S3 * S7
//                mul(&M[20 * h * h], &M[8 * h * h], &M[12 * h * h], h);  //M[20] = P5 = S1 * S5
//                mul(&M[21 * h * h], &M[11 * h * h], &M[7 * h * h], h);  //M[21] = P6 = S4 * B22
//                mul(&M[22 * h * h], &M[3 * h * h], &M[15 * h * h], h);  //M[22] = P7 = A22 * S8
//            }
//
//            //23-24: T1, T2
////#pragma omp task
////            {
//            add(&M[23 * h * h], &M[16 * h * h], &M[17 * h * h], h); //M[23] = T1 = P1 + P2
//            add(&M[24 * h * h], &M[23 * h * h], &M[19 * h * h], h); //M[24] = T2 = T1 + P4
//            //25-28: �������� �������������� ������� C
//            add(&M[25 * h * h], &M[17 * h * h], &M[18 * h * h], h); //M[25] = C11 = P2 + P3
//            add(&M[26 * h * h], &M[23 * h * h], &M[20 * h * h], h); //M[26] = C12 = T1 + P5
//            add(&M[26 * h * h], &M[26 * h * h], &M[21 * h * h], h); //M[26] = C12 += P6
//            sub(&M[27 * h * h], &M[24 * h * h], &M[22 * h * h], h); //M[27] = C21 = T2 - P7
//            add(&M[28 * h * h], &M[24 * h * h], &M[20 * h * h], h); //M[28] = C22 = T2 + P5
////            }
//#pragma omp taskwait
//        }
//        //�������� ���������
//        copyback_test(c, 0, 0, &M[25 * h * h], n);           //C11 = M[25]
//        copyback_test(c, 0, h, &M[26 * h * h], n);           //C12 = M[26]
//        copyback_test(c, h, 0, &M[27 * h * h], n);           //C21 = M[27]
//        copyback_test(c, h, h, &M[28 * h * h], n);           //C22 = M[28]
//
//        delete[]M;
//    }
//}
//
//int main()
//{
//    int n = 1024;
//
//
//    int** test1 = new int* [n];
//    for (int i = 0; i < n; ++i) {
//        test1[i] = new int[n];
//    }
//
//    int** test2 = new int* [n];
//    for (int i = 0; i < n; ++i) {
//        test2[i] = new int[n];
//    }
//
//    int** ctest = new int* [n];
//    for (int i = 0; i < n; ++i) {
//        ctest[i] = new int[n];
//    }
//
//    int** ctesto = new int* [n];
//    for (int i = 0; i < n; ++i) {
//        ctesto[i] = new int[n];
//    }
//
//
//    for (int i = 0; i < n; ++i)
//        for (int j = 0; j < n; ++j) {
//            test1[i][j] = rand() % 10000 + 1;
//            test2[i][j] = rand() % 10000 + 1;
//            ctest[i][j] = 0;
//            ctesto[i][j] = 0;
//        }
//
//    std::cout << "w/o openmp : " << std::endl;
//    for (int j = 1; j <= 1; ++j) {
//        std::cout << "try = " << j << std::endl;
//
//        {
//            Profiler Pr;
//            mul_test((int**)ctesto, (int**)test1, (int**)test2, n);
//        }
//    }
//    std::cout << "w openmp  :" << std::endl;
//
//
//    std::cout << "n =  " << n << std::endl;
//    for (int i = 1; i <= 8; ++i) {
//        for (int j = 1; j <= 1; ++j) {
//            std::cout << "threads =  " << i << "   try = " << j << std::endl;
//            {
//                auto t1 = omp_get_wtime();
//                Profiler Pr;
//                mul_test_optim((int**)ctest, (int**)test1, (int**)test2, n, i);
//                auto t2 = omp_get_wtime();
//                std::cout << t2 - t1 << std::endl;
//            }
//
//        }
//    }
//
//    return 0;
//}
//
