#include "Non_Opt_Pooling_C.h"

void mean_c(float* out,const float* patches,const TwoDimShape reduction_dims, const FourDimShape post_reduce_dims,const FiverrDimShape patches_Shape){
    //Tensor3xf out(post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]);
    int k_size,p_size,q_size;
    if(reduction_dims.dim0BeforeTrans==1&&reduction_dims.dim1BeforeTrans==2)
    {
        k_size=patches_Shape.dim3BeforeTrans;
        p_size=patches_Shape.dim1BeforeTrans;
        q_size=patches_Shape.dim2BeforeTrans;
    }
    else if (reduction_dims.dim0BeforeTrans==1&&reduction_dims.dim1BeforeTrans==3)
    {
        k_size=patches_Shape.dim2BeforeTrans;
        p_size=patches_Shape.dim1BeforeTrans;
        q_size=patches_Shape.dim3BeforeTrans;
    }
    else if (reduction_dims.dim0BeforeTrans==2&&reduction_dims.dim1BeforeTrans==3)
    {
        k_size=patches_Shape.dim1BeforeTrans;
        p_size=patches_Shape.dim2BeforeTrans;
        q_size=patches_Shape.dim3BeforeTrans;
    }
    for (int i = 0; i < patches_Shape.dim0BeforeTrans; i++)
    {
        for (int j = 0; j < patches_Shape.dim4BeforeTrans; j++)
        {
            for (int k = 0; k < k_size; k++)
            {
                float sum = 0;
                for (int p = 0; p < p_size; p++)
                {
                    for (int q = 0; q < q_size; q++)
                    {
                        sum+=patches[(((i*k_size+k)*p_size+p)*q_size+q)*patches_Shape.dim4BeforeTrans+j];
                    }
                }
                out[(i*k_size+k)*patches_Shape.dim4BeforeTrans+j] = sum / (p_size * q_size);
            }
        }
    }
}
void maximum_c(float* out,const float* patches,const TwoDimShape reduction_dims, const FourDimShape post_reduce_dims, const FiverrDimShape patches_Shape, Mask* mask){
    

    //Tensor3xf out(post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]);
    int k_size,p_size,q_size;
    if(reduction_dims.dim0BeforeTrans==1&&reduction_dims.dim1BeforeTrans==2)
    {
        k_size=patches_Shape.dim3BeforeTrans;
        p_size=patches_Shape.dim1BeforeTrans;
        q_size=patches_Shape.dim2BeforeTrans;
    }
    else if (reduction_dims.dim0BeforeTrans==1&&reduction_dims.dim1BeforeTrans==3)
    {
        k_size=patches_Shape.dim2BeforeTrans;
        p_size=patches_Shape.dim1BeforeTrans;
        q_size=patches_Shape.dim3BeforeTrans;
    }
    else if (reduction_dims.dim0BeforeTrans==2&&reduction_dims.dim1BeforeTrans==3)
    {
        k_size=patches_Shape.dim1BeforeTrans;
        p_size=patches_Shape.dim2BeforeTrans;
        q_size=patches_Shape.dim3BeforeTrans;
    }
    for (int i = 0; i < patches_Shape.dim0BeforeTrans; i++)
    {
        for (int j = 0; j < patches_Shape.dim4BeforeTrans; j++)
        {
            for (int k = 0; k < k_size; k++)
            {
                int p_key = 0; int q_key = 0;
                for (int p = 0; p < p_size; p++)
                {
                    for (int q = 0; q < q_size; q++)
                    {
                        if (patches[(((i*k_size+k)*p_size+p)*q_size+q)*patches_Shape.dim4BeforeTrans+j] > patches[(((i*k_size+k)*p_size+p_key)*q_size+q_key)*patches_Shape.dim4BeforeTrans+j]){ p_key = p; q_key = q; }
                    }
                }
                out[(i*k_size+k)*patches_Shape.dim4BeforeTrans+j] = patches[(((i*k_size+k)*p_size+p_key)*q_size+q_key)*patches_Shape.dim4BeforeTrans+j];
                mask[(i*k_size+k)*patches_Shape.dim4BeforeTrans+j].x = p_key;
                mask[(i*k_size+k)*patches_Shape.dim4BeforeTrans+j].y = q_key;
                
            }
        }
    }
}
