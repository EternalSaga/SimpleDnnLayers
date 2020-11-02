#include "PoolingLayer.hpp"
#include "TestUtils.hpp"
#define BOOST_TEST_MODULE LayerTest
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>


#include <iostream>
#include <cassert>
using namespace RLDNN;
BOOST_AUTO_TEST_CASE(PoolingEigenTest)
{   
    Tensor4xf inputX (2, 3, 2, 2);
    Tensor4xf expectForward (2, 3, 2, 2);
    Tensor4xf expectBackward (2, 3, 2, 2);
 
    inputX.setValues({{{{ 0.98632133 ,-1.5290705 },{-1.3209094 , -0.6300095 }},
{{ 0.01976758 , 0.4253078 },{ 0.22596048 , 0.37487507}},{{ 1.0715663 ,  0.08144144},{-0.11737227 , 0.31650934}}},
{{{ 0.33503392 , 2.796556  },{-1.2349964 ,  0.07790577}},{{-0.7392358 , -1.6891558 },{ 0.6077014 ,  0.5915958 }},
  {{-2.4391599 , -0.40140626},{-0.48190227 -0.3232832 }}}});
    expectForward.setValues({{{{0.986321,0.425308}}},{{{0.607701,2.79656}}}});
    expectBackward.setValues({{{{0.986321,0},{0,0}},{{0,0.425308},{0,0}},{{0,0},{0,0}}},{{{0,2.79656},{0,0}},{{0,0},{0.607701,0}},{{0,0},{0,0}}}});
   
    PoolingLayer<Tensor4xf, Device::CPU> p1{ TensorsWithNames<Tensor4xf> { {"x", inputX}},std::vector<int> {2,2},std::vector<int> {2,2},max };
    Tensor4xf out1 = p1.forwardImpl(outT);
    TensorsWithNames<Tensor4xf> out2=p1.backwardImpl(Eigen::DSizes<int, 4> {2,3,2,2},out1,outT);
    Tensor4xf dout=out2.at("dX");
	  bool isSame = RLDNN::TEST::tensorIsApprox2(out1, expectForward);
    bool isSame2 = RLDNN::TEST::tensorIsApprox2(dout, expectBackward);
   
    PoolingLayer<Tensor4xf, Device::CPU> p2{ TensorsWithNames<Tensor4xf> { {"x", inputX}},std::vector<int> {2,2},std::vector<int> {2,2},avg };
    Tensor4xf out11 = p2.forwardImpl(outT);
    TensorsWithNames<Tensor4xf> out22=p2.backwardImpl(Eigen::DSizes<int, 4> {2,3,2,2},out11,outT);
    Tensor4xf doutt=out22.at("dX");
}


