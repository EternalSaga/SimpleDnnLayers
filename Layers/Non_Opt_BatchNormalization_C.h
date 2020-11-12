#pragma once
#include<math.h>
typedef struct{
    int dim0BeforeTrans;
    int dim1BeforeTrans;
}TwoDimShape;
typedef struct{
    int dim0BeforeTrans;
}OneDimShape;

void mean_(float* out,int  reduction_dims, const float* post_reduce_dims,const TwoDimShape post_reduce_dims_shape);
void sub_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape);
void add_one(float* c,const float* a,const float* b,const OneDimShape a_Shape);
void div_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape);
void mul_one(float* c,const float* a,const float* b,const OneDimShape a_Shape);
void add_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape);
void mul_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape);
void sqrt_two(float* a,const TwoDimShape a_Shape);
void sqrt_one(float* a,const OneDimShape a_Shape);
void sum_( float* out,int add_dims,const float* post_add_dims,const TwoDimShape post_add_dims_shape);
void sum_de( float* out,int add_dims,const float* post_add_dims,const TwoDimShape post_add_dims_shape);
void div_one(float* c,const float* a,const float* b,const OneDimShape a_Shape);