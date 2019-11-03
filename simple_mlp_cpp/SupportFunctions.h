#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
namespace RLDNN {
using namespace Eigen;
using MatrixXfRow = Matrix<float, Dynamic, Dynamic, RowMajor>;
MatrixXfRow sigmoid(const MatrixXfRow& x);
//x is in mini-batch(it has multiple rows, each row for one result)
MatrixXfRow softmax(const MatrixXfRow& x);
MatrixXfRow sigmoidGrad(const MatrixXfRow& x);
//The ground truth is a one-hot vector
//All inputs are in mini-batch
float crossEntropyError(const MatrixXfRow& y, const MatrixXfRow& truth);
}  // namespace RLDNN