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
using namespace std::literals;
#define OP_CONCEPT_ERR "This class does not has forward or backward function."

enum Device { CPU, CUDA };

template <typename Precision, size_t Rank>
using OpInOutType = std::map<std::string_view, Tensor<Precision, Rank>>;

auto hasForward = hana::is_valid(
    [](auto&& x) -> decltype(x.forward(OpInOutType<float, 4>())) {});
auto hasBackward =
    hana::is_valid([](auto&& x) -> decltype(x.backward(Tensor<float, 4>())) {});
template <typename Op>
struct OpValidation {
  constexpr static auto valid =
      decltype(hasForward(Op{}))::value && decltype(hasBackward(Op{}))::value;
};

}  // namespace RLDNN
