#pragma once
#include <Eigen/Core>
#include <boost/hana.hpp>
#include <map>
#include <string_view>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
namespace RLDNN {
using namespace Eigen;
namespace hana = boost::hana;
enum Device { CPU, CUDA };

auto hasForward = hana::is_valid(
    [](auto&& x)
        -> decltype(x.forward(std::map<std::string_view, Tensor<float, 4>>())) {});

// template <typename Precision, size_t Rank>
// auto hasBackward = hana::is_valid(
//    [](auto&& x) -> decltype(x.backward(const Tensor<Precision, Rank>&)) {});

}  // namespace RLDNN
