#pragma once

#ifdef USE_MKL
#include <mkl.h>
#endif

void add_bias(float *output, float *biases, int batch, int n, int size);

void simplifiedSgemm(CBLAS_LAYOUT layout, float* A, const CBLAS_TRANSPOSE transA, float* B, const CBLAS_TRANSPOSE transB, float* C, const int m, const int n, const int k);