#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
namespace RLDNN {
using namespace Eigen;
using MatrixXfRow = Matrix<float, Dynamic, Dynamic, RowMajor>;
MatrixXfRow sigmoid(const MatrixXfRow& x);
MatrixXfRow softmax(const MatrixXfRow& x);
}  // namespace RLDNN