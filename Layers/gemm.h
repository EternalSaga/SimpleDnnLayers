#pragma once

//Compute C = ALPHA * A * B + BETA * C
/*
**  输入： A,B,C   输入矩阵（一维数组格式，按行存储，所有行并成一行）
**        ALPHA   系数
**        BETA    系数
**        TA,TB   是否需要对A,B做转置操作，是为1,否为0
**        M       A,C的行数
**        N       B,C的列数
**        K       A的列数，B的行数
**        lda     A的列数（不做转置）或者行数（做转置）
**        ldb     B的列数（不做转置）或者行数（做转置）
**        ldc     C的列数
*/
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc);