#include "matrixSupport.h"


void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i, j, b;
    int count=0;
    for (b = 0; b < batch; ++b)
    {
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                output[(b * n + i) * size + j] += biases[i];
                count++;
            }
        }
    }
}


void simplifiedSgemm(CBLAS_LAYOUT layout,const float* A, CBLAS_TRANSPOSE transA,const float* B, CBLAS_TRANSPOSE transB, float* C, const int m, const int n, const int k){
    layout = CblasRowMajor;

	int lda, ldb, ldc;
    
    if (transA == CblasNoTrans) {
        if (layout == CblasRowMajor)
        {
            lda = k;
        }
        else {
            lda = m;
        }
    }
    else {
        if (layout == CblasRowMajor)
        {
            lda = m;
        }
        else {
            lda = k;
        }
    }
    if (transB == CblasNoTrans) {
        if (layout == CblasRowMajor) {
            ldb = n;
        }
        else
        {
            ldb = k;
        }
    }
    else {
        if (layout == CblasRowMajor) {
            ldb = k;
        }
        else
        {
            ldb = n;
        }
    }
    if (layout == CblasRowMajor) {
        ldc = n;
    }
    else
    {
        ldc = m;
    }

	cblas_sgemm(layout, transA, transB, m, n, k, 1.f,
		A, lda, B, ldb, 0.f, C, ldc);
}