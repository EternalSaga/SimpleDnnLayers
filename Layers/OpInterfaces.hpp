#pragma once
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

template <typename Precision>
using Tensor4D = Eigen::Tensor<Precision, 4>;

template <typename Precision,
          size_t Rank,
          StorageOptions stgOpt = Eigen::ColMajor>
using TensorsWithNames = std::map<std::string_view, Tensor<Precision, Rank, stgOpt>>;

template <typename LayerImpl,
          typename Precision,
          size_t Rank,
          Device dev = Device::CPU,
          StorageOptions stgOpt = Eigen::ColMajor>
class LayerInterface
{
public:
    Tensor<Precision, Rank> forward(
        const TensorsWithNames<Precision, Rank, stgOpt> &inputs)
    {
        return static_cast<LayerImpl *>(this)->forwardImpl(inputs);
    }
    TensorsWithNames<Precision, Rank, stgOpt> backward(
        const Tensor<Precision, Rank> &inputD)
    {
        return static_cast<LayerImpl *>(this)->backwardImpl(inputD);
    }
};

} // namespace RLDNN