#pragma once
typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
}TwoDimShape;

typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
    int dim2BeforeTrans;
    int dim3BeforeTrans;
}FourDimShape;
typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
    int dim2BeforeTrans;
    int dim3BeforeTrans;
    int dim4BeforeTrans;
}FiverrDimShape;
typedef struct{
    int x;
    int y;
}Mask;
void mean_c(float* out,const float* patches,const TwoDimShape reduction_dims, const FourDimShape post_reduce_dims,const FiverrDimShape patches_Shape);
void maximum_c(float* out,const float* patches,const TwoDimShape reduction_dims, const FourDimShape post_reduce_dims, const FiverrDimShape patches_Shape, Mask* mask);
