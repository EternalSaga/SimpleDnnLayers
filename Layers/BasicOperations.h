#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
namespace RLDNN {
	using namespace Eigen;
template<size_t Rank1, size_t Rank2>
class ReLU {
 public:
  explicit ReLU()=default;
  ~ReLU()=default;
  decltype<auto> forward(const Tensor<float, Rank1>& x) {

  }
  decltype<auto> backward(const Tensor<float, Rank2> dout) {

  }
 private:
  std::vector<size_t> mask;
};

}