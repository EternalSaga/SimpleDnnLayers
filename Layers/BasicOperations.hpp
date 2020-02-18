#pragma once
#include "OpInterfaces.hpp"
namespace RLDNN
{

template <typename TensorType, Device dev>
class MultiplyLayer : public LayerInterface<MultiplyLayer<TensorType,dev>, TensorType,dev>
{
public:
  MultiplyLayer() = default;
  ~MultiplyLayer() = default;
  TensorType forwardImplImpl(const TensorsWithNames<TensorType> &args)
  {
    this->x = args.at("x");
    this->y = args.at("y");
    return x * y;
  }
  TensorsWithNames<TensorType> backwardImpl(const TensorType &dout)
  {
    std::map<std::string_view, TensorType> gradient;
    gradient["dx"] = dout * this->y;
    gradient["dy"] = dout * this->x;
    return gradient;
  }

private:
  TensorType x;
  TensorType y;
};

template <typename TensorType,
          Device dev>
class AddLayer : public LayerInterface<AddLayer<TensorType,dev>, TensorType,dev>
{
public:
  AddLayer() = default;
  ~AddLayer() = default;
  TensorType forwardImplImpl(
      const TensorsWithNames<TensorType> &inputs)
  {
    return inputs.at("x") + inputs.at("y");
  }
  TensorsWithNames<TensorType> backwardImpl(
      const TensorType &inputD)
  {

    return std::map{{"dx", inputD}, {"dy", inputD}};
  }
};

} // namespace RLDNN