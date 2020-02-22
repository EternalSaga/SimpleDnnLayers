#define BOOST_TEST_MODULE LayerTest
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <cassert>
#include "Layers/AffineLayer.hpp"
#include "TestUtils.hpp"
using namespace RLDNN;
using namespace RLDNN::TEST;
using tf2 = Eigen::Tensor<float, 2>;
tf2 inputX(5, 4);
tf2 weight(4, 3);
tf2 bias(5, 3);

tf2 expectForward(5, 3);
tf2 expectBackward(5, 4);
BOOST_AUTO_TEST_CASE(AffineEigenTest)
{
    inputX.setValues({ {0.49506637,0.33573946,0.74770531,0.60383894},
		{0.91835035,0.39440946,0.70142903,0.25944527},
	{0.91406385,0.61399933,0.70429428,0.6126195},
	{0.45152032,0.0276012,0.18655131,0.53795353},
	{0.9274986,0.55114785,0.81956188,0.06092862} });

	weight.setValues({ {0.59764962,0.69816611, 0.94761363},
		{0.82496823, 0.12595675, 0.22971291},
	{0.6888013,  0.30468398, 0.27121961},
	{0.11886512 ,0.1054455,  0.99006313} });

	bias.setValues({ {0.99862834, 0.92811981, 0.11160123},
		{0.27830395, 0.65270424, 0.11731532},
	{0.79530787, 0.44176877 ,0.14984584},
	{0.70204698, 0.69102719 ,0.9958344},
	{0.43664909, 0.71602569 ,0.15575331} });

	expectForward.setValues({ {2.15827473 ,1.60753295, 1.45848757},
		{1.66651519 ,1.58461539, 1.52526608},
	{2.40606563 ,1.43645968 ,1.95461917},
	{1.18710874, 1.12330391, 2.01324588},
	{2.01740533, 1.6891266 , 1.44387384} });

	expectBackward.setValues({ {3.79429979, 2.31802113 ,2.37198241 ,1.87004547},
		{3.54767986 ,1.9247884 , 2.04438683, 1.87529081},
	{4.29309544 ,2.61486076 ,2.62509844 ,2.37266186},
	{3.40150704 ,1.58328328, 1.70596652 ,2.25279368},
	{3.753227 ,  2.20872866 ,2.29584813 ,1.84743618} });


	TensorsWithNames<tf2> argWB{ { "weight" ,weight },{"bias",bias} };
	AffineLayer<tf2, Device::CPU> al{ argWB };
	tf2 outputX = al.forward(TensorsWithNames<tf2>{ {"x", inputX}});
	bool isSame =  tensorIsApprox(outputX, expectForward);
    BOOST_TEST(isSame);
    auto backwardResult = al.backward(outputX);
	isSame = tensorIsApprox(expectBackward, backwardResult.at("dx"));
    BOOST_TEST(isSame);

}