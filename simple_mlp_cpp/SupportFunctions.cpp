#include "SupportFunctions.h"

namespace RLDNN {
MatrixXfRow sigmoid(const MatrixXfRow& x) {
  return 1.f / (1.f + (-x).exp().array());
}
MatrixXfRow softmax(const MatrixXfRow& x) {
  auto y = x.array() - x.maxCoeff();
  return y.exp().array() / y.exp().array().sum();
}
}


