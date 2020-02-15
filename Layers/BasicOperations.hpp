#pragma once
#include "OpInterfaces.hpp"
namespace RLDNN
{

template <typename Precision, size_t Rank>
class MultiplyLayer : public LayerInterface<MultiplyLayer<Precision, Rank>, Precision, Rank>
{
public:
  MultiplyLayer() = default;
  ~MultiplyLayer() = default;
  Tensor<Precision, Rank> forwardImplImpl(const TensorsWithNames<Precision, Rank> &args)
  {
    this->x = args.at("x");
    this->y = args.at("y");
    return x * y;
  }
  TensorsWithNames<Precision, Rank> backwardImpl(const Tensor<Precision, Rank> &dout)
  {
    std::map<std::string_view, Tensor<Precision, Rank>> gradient;
    gradient["dx"] = dout * this->y;
    gradient["dy"] = dout * this->x;
    return gradient;
  }

private:
  Tensor<Precision, Rank> x;
  Tensor<Precision, Rank> y;
};

template <typename Precision,
          size_t Rank>
class AddLayer : public LayerInterface<AddLayer<Precision, Rank>, Precision, Rank>
{
public:
  AddLayer() = default;
  ~AddLayer() = default;
  Tensor<Precision, Rank> forwardImplImpl(
      const TensorsWithNames<Precision, Rank> &inputs)
  {
    return inputs.at("x") + inputs.at("y");
  }
  TensorsWithNames<Precision, Rank> backwardImpl(
      const Tensor<Precision, Rank> &inputD)
  {

    return std::map{{"dx", inputD}, {"dy", inputD}};
  }
};

} // namespace RLDNN