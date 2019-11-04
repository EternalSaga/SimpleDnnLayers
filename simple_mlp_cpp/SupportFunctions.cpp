#include "SupportFunctions.h"
#include <iostream>
namespace RLDNN {
MatrixXfRow sigmoid(const MatrixXfRow& x) {
  return 1.f / (1.f + ((0 - x.array()).exp()).array());
}
MatrixXfRow softmax(const MatrixXfRow& x) {
  // for mini batch
  if (x.rows() > 1) {
    MatrixXfRow y = x.colwise() - x.rowwise().maxCoeff();

    return y.array().exp().colwise() / y.array().exp().rowwise().sum().array();
  } else {
    // for a single result
    auto y(x.array() - x.maxCoeff());
    return y.exp().array() / y.exp().array().sum();
  }
}

MatrixXfRow sigmoidGrad(const MatrixXfRow& x) {
  return (1.0f - sigmoid(x).array()).array() * sigmoid(x).array();
}

float crossEntropyError(const MatrixXfRow& y, const MatrixXfRow& truth) {
  auto batchSize{y.rows()};
  assert(y.cols() == truth.cols() && y.rows() == truth.rows());
  return (0 - (truth.array() * ((y.array() + 1e-7).log())).sum()) / batchSize;
}

}  // namespace RLDNN
