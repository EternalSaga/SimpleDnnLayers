
#include "MlpNet.h"
namespace RLDNN {
MlpNet::MlpNet(int32_t inputSize,
               int32_t hiddenSize,
               int32_t outputSize,
               float weightInitStd) {
  params["W1"] = weightInitStd * MatrixXfRow(inputSize, hiddenSize).setRandom();
  params["b1"] = MatrixXfRow(hiddenSize, 1).setZero();
  params["W2"] = weightInitStd * MatrixXfRow(hiddenSize, outputSize).setRandom();
  params["b2"] = MatrixXfRow(outputSize, 1).setZero();
}
}  // namespace RLDNN