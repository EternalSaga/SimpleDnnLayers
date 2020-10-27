#include "Non_Opt_BatchNormalization_C.h"
Tensor1xf mean(Eigen::array<int, 1>  reduction_dims, Tensor2xf post_reduce_dims){
    Tensor1xf out(post_reduce_dims.dimension(1-reduction_dims[0]));
    for (int i = 0; i < post_reduce_dims.dimension(1-reduction_dims[0]); i++)
    {
        float sum = 0;
        for (int j = 0; j < post_reduce_dims.dimension(reduction_dims[0]); j++)
        {
            sum+=post_reduce_dims(i, j);         
        }
        out(i)= sum / post_reduce_dims.dimension(reduction_dims[0]);    
    }
    return out;
}
Tensor2xf sub(Tensor2xf a,Tensor2xf b)
{
    Tensor2xf out(a.dimensions());
    for(int i=0;i<a.dimension(0);i++)
    {
        for(int j=0;j<a.dimension(1);j++)
        {
            out(i,j)=a(i,j)-b(i,j);
        }
    }
    return out;
}
Tensor1xf add(Tensor1xf a,Tensor1xf b)
{
    Tensor1xf out(a.dimensions());
    for(int i=0;i<a.dimension(0);i++)
    { 
        out(i)=a(i)+b(i);
    }
    return out;
}
Tensor2xf div(Tensor2xf a,Tensor2xf b)
{
    Tensor2xf out(a.dimensions());
    for(int i=0;i<a.dimension(0);i++)
    {
        for(int j=0;j<a.dimension(1);j++)
        {
            out(i,j)=a(i,j)/b(i,j);
        }
    }
    return out;
}
Tensor1xf mul(Tensor1xf a,Tensor1xf b)
{
    Tensor1xf out(a.dimensions());
    for(int i=0;i<a.dimension(0);i++)
    { 
        out(i)=a(i)*b(i);
    }
    return out;
}
Tensor2xf add(Tensor2xf a,Tensor2xf b)
{
    Tensor2xf out(a.dimensions());
    for(int i=0;i<a.dimension(0);i++)
    {
        for(int j=0;j<a.dimension(1);j++)
        {
            out(i,j)=a(i,j)+b(i,j);
        }
    }
    return out;
}
Tensor2xf mul(Tensor2xf a,Tensor2xf b)
{
    Tensor2xf out(a.dimensions());
    for(int i=0;i<a.dimension(0);i++)
    {
        for(int j=0;j<a.dimension(1);j++)
        {
            out(i,j)=a(i,j)*b(i,j);
        }
    }
    return out;
}
Tensor2xf sqrt_(Tensor2xf a)
{
    for(int i=0;i<a.dimension(0);i++)
    {
        for(int j=0;j<a.dimension(1);j++)
        {
            a(i,j)=sqrt(a(i,j));
        }
    }
    return a;
}

Tensor1xf sqrt_(Tensor1xf a)
{
    for(int i=0;i<a.dimension(0);i++)
    {
        a(i)=sqrt(a(i));
    }
    return a;
}
Tensor1xf sum(Eigen::array<int, 1> add_dims,Tensor2xf post_add_dims)
{
    Tensor1xf out(post_add_dims.dimension(1-add_dims[0]));
    for (int i = 0; i < post_add_dims.dimension(1-add_dims[0]); i++)
    {
        float sum = 0;
        for (int j = 0; j < post_add_dims.dimension(add_dims[0]); j++)
        {
            sum+=post_add_dims(i, j);         
        }
        out(i)= sum;    
    }
    return out;
}
Tensor1xf div(Tensor1xf a,Tensor1xf b)
{
    Tensor1xf out(a.dimensions());
    for(int i=0;i<a.dimension(0);i++)
    {
        out(i)=a(i)/b(i);
    }
    return out;
}
