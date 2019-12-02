#pragma once
#include <Eigen/Core>
#include <boost/hana.hpp>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string_view>
#include <map>
namespace RLDNN {
using namespace Eigen;
namespace hana = boost::hana;
enum Device { CPU, CUDA };
auto has_toString =
    hana::is_valid([](auto&& obj) -> decltype(obj.toString()) {});
template <typename Precision, size_t Rank>
auto hasForward = hana::is_valid(
    [](auto&& x)
        -> decltype(x.forward(std::map<std::string_view,Tensor<Precision, Rank>>)) {});

//template <typename Precision, size_t Rank>
//auto hasBackward = hana::is_valid(
//    [](auto&& x) -> decltype(x.backward(const Tensor<Precision, Rank>&)) {});


}  // namespace RLDNN
