#include "BasicOperations.h"

#include <iostream>
int main() {
  using namespace RLDNN;
  MultiplyLayer<float, 4> mt1{};
  Tensor<float, 4> a(2, 2, 2, 2);
  Tensor<float, 4> b(2, 2, 2, 2);
  a.setRandom();
  b.setRandom();
  auto result = mt1.forward(OpInOutType<float,4>{{"x", a}, {"y", b}});
  auto derivedResult = mt1.backward(result);
  std::cout << result << std::endl;
  std::cout << derivedResult.at("dx") << std::endl;
  std::cout << derivedResult.at("dy") << std::endl;
  return 0;
}