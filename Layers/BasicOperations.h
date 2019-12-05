#pragma once
#include "OpInterfaces.h"
namespace RLDNN {

template <typename Precision, size_t Rank>
class MultiplyLayer {
 public:
  MultiplyLayer() = default;
  ~MultiplyLayer() = default;
  Tensor<Precision, Rank> forward(const OpInOutType<Precision, Rank>& args) {
    this->x = args.at("x");
    this->y = args.at("y");
    return x * y;
  }
  OpInOutType<Precision, Rank> backward(const Tensor<Precision, Rank>& dout) {
    std::map<std::string_view, Tensor<Precision, Rank>> gradient;
    gradient["dx"] = dout * this->y;
    gradient["dy"] = dout * this->x;
    return gradient;
  }

 private:
  Tensor<Precision, Rank> x;
  Tensor<Precision, Rank> y;
};

static_assert(OpValidation<MultiplyLayer<float, 4>>::valid, OP_CONCEPT_ERR);

template <typename Precision, size_t Rank>
class AddLayer {
 public:
  AddLayer() = default;
  ~AddLayer() = default;
  Tensor<Precision, Rank> forward(const OpInOutType<Precision, Rank>& args) {
    return args["x"] + args["y"];
  }
  OpInOutType<Precision, Rank> backward(const Tensor<Precision, Rank>& dout) {
    return std::map{{"dx", dout}, {"dy", dout}};
  }
};
static_assert(OpValidation<AddLayer<float, 4>>::valid, OP_CONCEPT_ERR);


}  // namespace RLDNN