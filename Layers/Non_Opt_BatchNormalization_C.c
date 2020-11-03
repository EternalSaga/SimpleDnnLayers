
#include "Non_Opt_BatchNormalization_C.h"
void mean_(float* out,int  reduction_dims, const float* post_reduce_dims,const TwoDimShape post_reduce_dims_shape){
    //Tensor1xf out(post_reduce_dims.dimension(1-reduction_dims[0]));
    int i_size,j_size;
    if(reduction_dims==0)
    {
        i_size=post_reduce_dims_shape.dim1BeforeTrans;
        j_size=post_reduce_dims_shape.dim0BeforeTrans;
    }else if(reduction_dims==1)
    {
        i_size=post_reduce_dims_shape.dim0BeforeTrans;
        j_size=post_reduce_dims_shape.dim1BeforeTrans;
    }
    for (int i = 0; i < i_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < j_size; j++)
        {
            sum+=post_reduce_dims[i*j_size+j];         
        }
        out[i]= sum / j_size;    
    }
}
void sub_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape)
{
    //Tensor2xf out(a.dimensions());
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    {
        for(int j=0;j<a_Shape.dim1BeforeTrans;j++)
        {
            c[i*a_Shape.dim1BeforeTrans+j]=a[i*a_Shape.dim1BeforeTrans+j]-b[i*a_Shape.dim1BeforeTrans+j];
        }
    }
}
void add_one(float* c,const float* a,const float* b,const OneDimShape a_Shape)
{
    //Tensor1xf out(a.dimensions());
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    { 
        c[i]=a[i]+b[i];
    }
}
void div_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape)
{
    //Tensor2xf out(a.dimensions());
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    {
        for(int j=0;j<a_Shape.dim1BeforeTrans;j++)
        {
            c[i*a_Shape.dim1BeforeTrans+j]=a[i*a_Shape.dim1BeforeTrans+j]/b[i*a_Shape.dim1BeforeTrans+j];
        }
    }
}
void mul_one(float* c,const float* a,const float* b,const OneDimShape a_Shape)
{
    //Tensor1xf out(a.dimensions());
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    { 
        c[i]=a[i]*b[i];
    }
}
void add_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape)
{
    //Tensor2xf out(a.dimensions());
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    {
        for(int j=0;j<a_Shape.dim1BeforeTrans;j++)
        {
            c[i*a_Shape.dim1BeforeTrans+j]=a[i*a_Shape.dim1BeforeTrans+j]+b[i*a_Shape.dim1BeforeTrans+j];
        }
    }

}
void mul_two(float* c,const float* a,const float* b,const TwoDimShape a_Shape)
{
    //Tensor2xf out(a.dimensions());
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    {
        for(int j=0;j<a_Shape.dim1BeforeTrans;j++)
        {
            c[i*a_Shape.dim1BeforeTrans+j]=a[i*a_Shape.dim1BeforeTrans+j]*b[i*a_Shape.dim1BeforeTrans+j];
        }
    }
}
void sqrt_two(float* a,const TwoDimShape a_Shape)
{
    
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    {
        for(int j=0;j<a_Shape.dim1BeforeTrans;j++)
        {
            a[i*a_Shape.dim1BeforeTrans+j]=sqrt(a[i*a_Shape.dim1BeforeTrans+j]);
        }
    }
}

void sqrt_one(float* a,const OneDimShape a_Shape)
{
    
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    {
        
        
        a[i]=sqrt(a[i]);
        
    }
}
void sum_( float* out,int add_dims,const float* post_add_dims,const TwoDimShape post_add_dims_shape)
{
    //Tensor1xf out(post_add_dims.dimension(1-add_dims[0]));
    int i_size,j_size;
    if(add_dims==0)
    {
        i_size=post_add_dims_shape.dim1BeforeTrans;
        j_size=post_add_dims_shape.dim0BeforeTrans;
    }else if(add_dims==1)
    {
        i_size=post_add_dims_shape.dim0BeforeTrans;
        j_size=post_add_dims_shape.dim1BeforeTrans;
    }
    for (int i = 0; i < i_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < j_size; j++)
        {
            sum+=post_add_dims[i*j_size+j];         
        }
        out[i]= sum;    
    }

}
void sum_de( float* out,int add_dims,const float* post_add_dims,const TwoDimShape post_add_dims_shape)
{
    //Tensor1xf out(post_add_dims.dimension(1-add_dims[0]));
    int i_size,j_size;
    if(add_dims==0)
    {
        i_size=post_add_dims_shape.dim1BeforeTrans;
        j_size=post_add_dims_shape.dim0BeforeTrans;
    }else if(add_dims==1)
    {
        i_size=post_add_dims_shape.dim0BeforeTrans;
        j_size=post_add_dims_shape.dim1BeforeTrans;
    }
    for (int i = 0; i < i_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < j_size; j++)
        {
            sum+=post_add_dims[i*j_size+j];         
        }
        out[i]= -sum;    
    }

}
void div_one(float* c,const float* a,const float* b,const OneDimShape a_Shape)
{
    //Tensor1xf out(a.dimensions());
    for(int i=0;i<a_Shape.dim0BeforeTrans;i++)
    {
        
        
        c[i]=a[i]/b[i];
        
    }

}
