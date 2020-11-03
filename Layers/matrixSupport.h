#pragma once

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#define CBLAS_LAYOUT CBLAS_ORDER
#endif

// A    simlified sgemm used to compute C = A crossProduct B;
// C    is assumed as a zero matrix
//      IMPORTANT!! OP could be the transpoe operation!!!
// m    rows of the matrix op(A)
// n    columns of the matrix op(B)
// k    columns of the matrix op(A)
void simplifiedSgemm(CBLAS_LAYOUT layout,const float* A, CBLAS_TRANSPOSE transA,const float* B, CBLAS_TRANSPOSE transB, float* C, const int m, const int n, const int k);

void add_bias(float *output, float *biases, int batch, int n, int size);
