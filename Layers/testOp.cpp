//#include "BasicOperations.h"
#include <iostream>

#include "ActivationLayers.h"
int main() {
  using namespace RLDNN;
  RelULayer<float, 4> mt1{};
  Tensor<float, 4> a(2, 2, 2, 2);
  a.setRandom();
  Tensor<float, 4> out = mt1.forward(OpInOutType<float, 4>{{"x", a}});
  
  std::cout << out << std::endl;
  
  //Tensor<bool, 2> tbool(2, 2);
  //tbool.setValues({{true,true}, {false,false}});
  //std::cout << tbool.cast<float>() << std::endl;

  OpInOutType<float, 4> dout = mt1.backward(out);
  std::cout << dout.at("dx") << std::endl;
  return 0;
}
