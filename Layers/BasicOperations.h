#pragma once
#include "OpInterfaces.h"

namespace RLDNN {
template <typename Precision, size_t Rank>
class MultiplyLayer {
	
 public:
  MultiplyLayer()=default;
  ~MultiplyLayer()=default;
  Tensor<Precision, Rank> forward(
      const std::map<std::string_view,Tensor<Precision, Rank>>& args) {
    return args["x"] * args["y"];
  }
 private:
};

static_assert(decltype(hasForward<float,4>(MultiplyLayer<float,4>{}))::value,"This Op does not have forward function.");

}