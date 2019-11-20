#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include "SupportFunctions.h"
#include "load_mnist.h"
#include "testRLDNN.h"
namespace RLDNN {
class TrainMlp;
class MlpNet {
  friend class TrainMlp; 
  friend void TEST::testPredict();
  std::map<std::string, MatrixXfRow> params;
  std::map<std::string, MatrixXfRow> gradient(const MatrixXfRow& x,
                                              const MatrixXfRow& trueth);
  std::tuple<MatrixXfRow, MatrixXfRow, MatrixXfRow, MatrixXfRow> predict(
      const MatrixXfRow& x);

  float loss(const MatrixXfRow& x, const MatrixXfRow& truth);
  

 public:
  MlpNet(int32_t inputSize,
         int32_t hiddenSize,
         int32_t outputSize,
         float weightInitStd = 0.01);
  float getAccuracy(const MatrixXfRow& x, const MatrixXfRow& truth);
#ifndef NDEBUG
  MlpNet() = default;
#endif  // !NDEBUG
};
}  // namespace RLDNN
