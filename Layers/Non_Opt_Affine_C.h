#include "matrixSupport.h"

typedef struct {
    const float* x;
    const float* weight;
    const float* bias;
    const int m;
    const int n;
    const int k;
} ForwardArgs;

void affineForward(ForwardArgs fa, float* outdata);

typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
}TwoDimShape;

typedef struct {
    const float* inputD;
    const float* weight;
    const float* bias;
    const TwoDimShape inputDShape;
    const TwoDimShape weightShape;
    const int biasSize;
}BackwardArgs;

typedef struct {
    float* dx;
    float* dWeight;
    float* dBias;
    TwoDimShape dxShape;
} BackwardOut;

void affineBackward(BackwardArgs ba, BackwardOut bo);
