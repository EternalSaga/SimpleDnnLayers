#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <tuple>
#include "load_mnist.h"
#include "SupportFunctions.h"
namespace RLDNN {

class MlpNet {
  std::map<std::string, MatrixXfRow> params;
  std::map<std::string, MatrixXfRow> gradient(const MatrixXfRow& x,
                                              RowVectorXf& trueth);
  std::tuple<RowVectorXf, RowVectorXf, RowVectorXf, RowVectorXf> predict(
      const MatrixXfRow& x);

  float loss(const MatrixXfRow& x, const MatrixXfRow& truth);
 public:
  MlpNet(std::string mnistRootPath,
	  int32_t inputSize,
         int32_t hiddenSize,
         int32_t outputSize,
         float weightInitStd = 0.01);
  
  
};
}  // namespace RLDNN
