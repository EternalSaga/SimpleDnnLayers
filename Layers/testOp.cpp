//#include "BasicOperations.h"
#include "ActivationLayers.h"
#include <iostream>
int main()
{
  using namespace RLDNN;
  RelULayer<float, 4> mt1{};
  Tensor<float, 4> a(2, 2, 2, 2);
  a.setRandom();
  Tensor<float, 4> out = mt1.forward(OpInOutType<float,4>{{"x",a}});
  std::cout << out << std::endl;
  return 0;
}
