#pragma once
#include <Eigen/LU>
#include <Eigen/Core>
#include <map>
#include <string_view>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

namespace RLDNN
{
using namespace Eigen;

enum class Device
{
    CPU,
    CUDA,
    NON_OPTIMIZE
};

template <typename TensorType>
using TensorsWithNames = std::map<std::string_view, TensorType>;
template <typename LayerImpl,
          typename TensorType, Device dev>
class LayerInterface
{
    
public:

    static_assert(std::is_same_v<TensorType,Eigen::Tensor<typename TensorType::Scalar,TensorType::NumDimensions,
    TensorType::Layout>>,"Arguement TensorType is not a Eigen::Tensor type");
    TensorType forward(
        const TensorsWithNames<TensorType> &inputs)
    {
        return static_cast<LayerImpl *>(this)->forwardImpl(inputs);
    }
    TensorsWithNames<TensorType> backward(
        const TensorType &inputD)
    {
        return static_cast<LayerImpl *>(this)->backwardImpl(inputD);
    }
};

} // namespace RLDNN