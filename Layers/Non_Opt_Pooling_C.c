#include "Non_Opt_Pooling_C.h"

Tensor3xf mean_c(Tensor5xf patches,Eigen::array<int, 2> reduction_dims, Eigen::DSizes<int, 4>post_reduce_dims){
    Tensor3xf out(post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]);
    for (int i = 0; i < patches.dimension(0); i++)
    {
        for (int j = 0; j < patches.dimension(4); j++)
        {
            for (int k = 0; k < patches.dimension(1 + 2 + 3 - reduction_dims[0] - reduction_dims[1]); k++)
            {
                float sum = 0;
                for (int p = 0; p < patches.dimension(reduction_dims[0]); p++)
                {
                    for (int q = 0; q < patches.dimension(reduction_dims[1]); q++)
                    {
                        sum+=patches(i, k,p, q,  j);
                    }
                }
                out(i, k, j) = sum / (patches.dimension(reduction_dims[0]) * patches.dimension(reduction_dims[1]));
            }
        }
    }
    return out;
}
Tensor3xf maximum_c(Tensor5xf patches, Eigen::array<int, 2> reduction_dims, Eigen::DSizes<int, 4>post_reduce_dims, Eigen::Tensor<Mask, 3,Eigen::RowMajor> &mask){
    
    Tensor3xf out(post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]);
    
    for (int i = 0; i < patches.dimension(0); i++)
    {
        for (int j = 0; j < patches.dimension(4); j++)
        {
            for (int k = 0; k < patches.dimension(1+2+3- reduction_dims[0]- reduction_dims[1]); k++)
            {
                int p_key = 0; int q_key = 0;
                for (int p = 0; p < patches.dimension(reduction_dims[0]); p++)
                {
                    for (int q = 0; q < patches.dimension(reduction_dims[1]); q++)
                    {
                        if (patches(i,k, p, q,  j) > patches(i,k,  p_key, q_key, j)){ p_key = p; q_key = q; }
                    }
                }
                out(i, k, j) = patches(i, k, p_key, q_key, j);
                mask(i, k, j) = Mask(p_key,q_key);    
            }
        }
    }
    return out;
}
