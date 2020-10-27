#pragma once
#include "OpInterfaces.hpp"

#include <math.h>
using Tensor1xf = Eigen::Tensor<float, 1,Eigen::RowMajor>;
using Tensor2xf = Eigen::Tensor<float, 2,Eigen::RowMajor>;
using Tensor4xf = Eigen::Tensor<float, 4,Eigen::RowMajor>;
Tensor1xf mean(Eigen::array<int, 1> reduction_dims, Tensor2xf post_reduce_dims);
Tensor2xf sub(Tensor2xf a,Tensor2xf b);
Tensor1xf add(Tensor1xf a,Tensor1xf b);
Tensor2xf div(Tensor2xf a,Tensor2xf b);
Tensor1xf mul(Tensor1xf a,Tensor1xf b);
Tensor2xf add(Tensor2xf a,Tensor2xf b);
Tensor2xf mul(Tensor2xf a,Tensor2xf b);
Tensor2xf sqrt_(Tensor2xf a);
Tensor1xf sqrt_(Tensor1xf a);
Tensor1xf sum(Eigen::array<int, 1> add_dims,Tensor2xf post_add_dims);
Tensor1xf div(Tensor1xf a,Tensor1xf b);
