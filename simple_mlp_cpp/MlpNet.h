#pragma once
#include <cstdint>
#include <utility>
#include <string>
#include <Eigen/Core>
#include <map>
namespace RLDNN {
	using namespace Eigen;
	using MatrixXfRow = Matrix<float, Dynamic, Dynamic, RowMajor>;
class MlpNet {
          std::map<std::string, MatrixXfRow>
              params;
 public:
  MlpNet(int32_t inputSize,
         int32_t hiddenSize,
         int32_t outputSize,
         float weightInitStd = 0.01);
};
}

