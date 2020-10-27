#pragma once
#include "OpInterfaces.hpp"
class Mask
{
public:
    int x, y;
    Mask(){}
    Mask(int x, int y)
    {
        this->x = x;
        this->y = y;
    }

};

using Tensor3xf = Eigen::Tensor<float, 3,Eigen::RowMajor>;
using Tensor4xf = Eigen::Tensor<float, 4,Eigen::RowMajor>;
using Tensor5xf = Eigen::Tensor<float, 5,Eigen::RowMajor>;

            
Tensor3xf mean_c(Tensor5xf patches, Eigen::array<int, 2> reduction_dims, Eigen::DSizes<int, 4>post_reduce_dims);
Tensor3xf maximum_c(Tensor5xf patches, Eigen::array<int, 2> reduction_dims, Eigen::DSizes<int, 4>post_reduce_dims, Eigen::Tensor<Mask, 3,Eigen::RowMajor> &mask);
