
#include "MlpNet.h"
namespace RLDNN {
std::map<std::string, MatrixXfRow> MlpNet::gradient(const MatrixXfRow& x,
                                                    RowVectorXf& trueth) {
  auto batchNum(x.rows());
  // forward
  auto [a1, z1, a2, y](predict(x));
  // backward
  std::map<std::string, MatrixXfRow> grads;
  auto dy((y - trueth) / batchNum);
  grads["W2"] = z1.transpose() * dy;
  grads["b2"] = dy.colwise().sum();
  auto da1(z1.transpose() * dy);
  auto dz1(sigmoidGrad(a1).array() * da1.array());
  grads["W1"] = x.transpose() * dz1.matrix();
  grads["b1"] = dz1.colwise().sum();
  return grads;
}
std::tuple<RowVectorXf, RowVectorXf, RowVectorXf, RowVectorXf> MlpNet::predict(
    const MatrixXfRow& x) {
  auto& W1(params["W1"]);
  auto& W2(params["W2"]);
  auto& b1(params["b1"]);
  auto& b2(params["b2"]);
  // forward
  auto a1(x * W1 + b1);
  auto z1(sigmoid(a1));
  auto a2(z1 * W2 + b2);
  auto y(softmax(a2));

  return std::make_tuple(a1, z1, a2, y);
}
MlpNet::MlpNet(std::string mnistRootPath,
               int32_t inputSize,
               int32_t hiddenSize,
               int32_t outputSize,
               float weightInitStd) {
  params["W1"] = weightInitStd * MatrixXfRow(inputSize, hiddenSize).setRandom();
  params["b1"] = MatrixXfRow(hiddenSize, 1).setZero();
  params["W2"] =
      weightInitStd * MatrixXfRow(hiddenSize, outputSize).setRandom();
  params["b2"] = MatrixXfRow(outputSize, 1).setZero();
  
}
float MlpNet::loss(const MatrixXfRow& x, const MatrixXfRow& truth) {
  auto [_1, _2, _3, y]{predict(x)};
  return crossEntropyError(y, truth);
}
}  // namespace RLDNN