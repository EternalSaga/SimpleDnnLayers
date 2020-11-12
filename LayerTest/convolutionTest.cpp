#include "ConvolutionLayer.hpp"
#define BOOST_TEST_MODULE LayerTest
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/property_tree/json_parser.hpp>


#include <cassert>
using namespace RLDNN;
BOOST_AUTO_TEST_CASE(ConvolutionEigenTest)
{   
    Tensor4xf inputX (2, 3, 2, 2);

    
    inputX.setValues({{{{ 0.9,1.5 },{-1.3 , -0.6 }},
{{ 0.1 , 0.4 },{ 0.2, 0.3}},{{ 1.0 ,  0.8},{-0.1 , 0.3}}},
{{{ 0.3 , 2.7  },{-1.2 ,  0.7}},{{-0.7 , -1.6},{ 0.6 ,  0.5 }},
  {{-2.4, -0.4},{-0.4, -0.3 }}}});

    Tensor4xf filter(1,2,2,2);
    filter.setValues(
{{{{ 1.0, 1.0},
   { 1.0 , 1.0}},

  {{1.0 ,1.0},
   { 1.0 ,1.0}}}});
 
    
    ConvolutionLayer<Tensor4xf, Device::CPU> p1{ filter,std::vector<int> {1,1} };
  
    Tensor4xf out1 = p1.forwardImpl(TensorsWithNames<Tensor4xf> { {"x", inputX}});
    
    TensorsWithNames<Tensor4xf> out2=p1.backwardImpl(out1);
    Tensor4xf dout=out2.at("dX");
 

  
}
