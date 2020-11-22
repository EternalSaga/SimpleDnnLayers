#pragma once
typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
    int dim2BeforeTrans;
    int dim3BeforeTrans;
    int dim4BeforeTrans;
}FiveDimShape;
typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
    int dim2BeforeTrans;
    int dim3BeforeTrans;
}FourDimShape;
typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
}TwoDimShape;
void for_dot(float* conv,const float* patches,const FiveDimShape patches_Shape,const float* filter,const FourDimShape filter_Shape);
void db_dot(float* dB,const float* dout,const FourDimShape dout_Shape);
void dw_dot(float* dW,const FourDimShape dW_Shape,const float* dout,const FourDimShape dout_Shape,const float* x,const FourDimShape x_Shape);
void filter_dot(float* flip_filter,const FourDimShape flip_filter_Shape,const float* filter,const FourDimShape filter_Shape);
void bac_dot(float* conv,const float* patches,const FiveDimShape patches_Shape,const float* flip_filter,const FourDimShape flip_filter_Shape);
void padding_dot(float* padding_x,const FourDimShape padding_x_Shape,const TwoDimShape in_Shape,const float* argX,const FourDimShape argX_Shape);
void convert(float* out,const FourDimShape out_Shape,const float* x,const FourDimShape x_Shape);
