#define BOOST_TEST_MODULE LayerTest
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <cassert>
#include <ActivationLayers.hpp>
#include "TestUtils.hpp"
using namespace RLDNN;
using namespace RLDNN::TEST;

BOOST_AUTO_TEST_CASE(SigmoidTest)
{
	using Tf2 = Eigen::Tensor<float, 2>;
	SigmoidLayer<Tf2> sml{};
	Tensor<float, 2> testData(4, 3);
	testData.setValues({{100, 200, 300}, {550, 200, 250}, {1.3, 4.5, 0.33}, {332, 423, 1.1}});
	std::cout << testData << std::endl;
	Tensor<float, 2> expected(4, 3);
	expected.setValues({{1., 1., 1.}, {1., 1., 1.}, {0.78583498, 0.98901306, 0.58175938}, {1., 1., 0.75026011}});
	Tensor<float, 2> output = sml.forward(TensorsWithNames<Tf2>{{"x", testData}});
	bool isSame = tensorIsApprox<Eigen::Tensor<float, 2>>(expected, output);
	BOOST_TEST(isSame);
	TensorsWithNames<Tf2> dx = sml.backward(output);
	Tensor<float, 2> expectDx(4, 3);
	expectDx.setValues({{0, 0, 0}, {0, 0, 0}, {0.13225472, 0.01074683, 0.14155102}, {0., 0., 0.14057614}});
	isSame = tensorIsApprox<Eigen::Tensor<float, 2>>(expectDx, dx.at("dx"));
	BOOST_TEST(isSame);
}

BOOST_AUTO_TEST_CASE(ReLUTest_C)
{
	using Tf2 = Eigen::Tensor<float, 2>;
	RelULayer<Tf2, Device::CPU> ReluC{};
	Tf2 testData(2, 3);
	testData.setValues({{-1, 0, 2}, {2, 1, -1}});
	Tf2 forwardExpected(2, 3);
	forwardExpected.setValues({{0, 0, 2}, {2, 1, 0}});
	auto forwardResult = ReluC.forward(TensorsWithNames<Tf2>{{"x", testData}});
	bool isSame = tensorIsApprox<Tf2>(forwardResult, forwardExpected);
	BOOST_TEST(isSame);
	auto backwardResult = ReluC.backward(forwardResult);
	Tf2 backwardExpect(2, 3);
	backwardExpect.setValues({{0, 0, 1}, {1, 1, 0}});
	isSame = tensorIsApprox<Tf2>(backwardExpect, backwardResult.at("dx"));
	BOOST_TEST(isSame);
}