#include "BatchNormalizationLayer.hpp"
#define BOOST_TEST_MODULE LayerTest
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/property_tree/json_parser.hpp>


#include <cassert>
using namespace RLDNN;
BOOST_AUTO_TEST_CASE(BatchNormalizationEigenTest)
{   
   
    Tensor4xf input (2, 3, 1, 1);
    input.setValues({{{{-0.2}},{{0.9}},{{-0.4}}},{{{-0.3}},{{-0.0}},{{-0.7}}}});
    
    Tensor1xf gamma(2);
    gamma.setValues({1.0,1.0});
    Tensor1xf beta(2);
    beta.setValues({0.0,0.0});
    Tensor1xf batch_Mean(2);
    batch_Mean.setValues({0.0,0.0});
    Tensor1xf batch_Var(2);
    batch_Var.setValues({1.0,1.0});
    
    BatchNormalizationLayer<Tensor4xf, Device::CPU> p1{gamma,beta, batch_Mean, batch_Var,0.0001,1};
  
    Tensor4xf out1 = p1.forwardImpl(TensorsWithNames<Tensor4xf> { {"x", input}},out);
  
    TensorsWithNames<Tensor4xf> out2=p1.backwardImpl(out1,out);
    Tensor4xf dout=out2.at("dX");
}
