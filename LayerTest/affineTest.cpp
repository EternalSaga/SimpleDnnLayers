#define BOOST_TEST_MODULE LayerTest
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <cassert>
#include "AffineLayer.hpp"
#include "TestUtils.hpp"
using namespace RLDNN;
using namespace RLDNN::TEST;

BOOST_AUTO_TEST_CASE(AffineEigenTest)
{
	using tf2 = Eigen::Tensor<float, 2>;
	tf2 inputX(5, 4);
	tf2 weight(4, 3);
	tf2 bias(1, 3);

	tf2 expectForward(5, 3);
	tf2 expectBackward(5, 4);
	inputX.setValues({{0.03765099, 0.97142534, 0.8405472, 0.44674567},
					  {0.68986554, 0.80415858, 0.20979938, 0.69946131},
					  {0.24729487, 0.80640938, 0.74582434, 0.14324305},
					  {0.80681086, 0.90040793, 0.51829794, 0.77721708},
					  {0.42278581, 0.34412754, 0.78025222, 0.5804827}});

	weight.setValues({{0.09216211, 0.49851056, 0.67359579},
					  {0.21330701, 0.91796532, 0.30249352},
					  {0.26688717, 0.27553562, 0.96643816},
					  {0.17035219, 0.75909068, 0.60404623}});

	bias.setValues({{0.75297194, 0.54182269, 0.70450569}});

	expectForward.setValues({{1.26408913, 2.02304805, 2.10590903},
							 {1.16323159, 2.2126794, 2.03771407},
							 {1.17122864, 1.71959327, 1.92233461},
							 {1.29012035, 2.50335746, 2.4907156},
							 {1.17246739, 1.72410931, 2.19809269}});

	expectBackward.setValues({{2.54354339, 2.76375086, 2.9300218, 3.02308367},
							  {2.58284554, 2.89568372, 2.8894482, 3.10865685},
							  {2.26005481, 2.40985204, 2.64421261, 2.66602756},
							  {3.04458588, 3.32661238, 3.44120331, 3.6245575},
							  {2.44816974, 2.49767687, 2.91229067, 2.83623729}});

	TensorsWithNames<tf2> argWB{{"weight", weight}, {"bias", bias}};
	AffineLayer<tf2, Device::CPU> al{argWB};
	tf2 outputX = al.forward(TensorsWithNames<tf2>{{"x", inputX}});
	bool isSame = tensorIsApprox(outputX, expectForward);
	BOOST_TEST(isSame);
	auto backwardResult = al.backward(outputX);
	isSame = tensorIsApprox(expectBackward, backwardResult.at("dx"));
	BOOST_TEST(isSame);
}

BOOST_AUTO_TEST_CASE(AffineCTest)
{
	using tf2 = Eigen::Tensor<float, 2, Eigen::RowMajor>;
	tf2 inputX(5, 4);
	tf2 weight(4, 3);
	tf2 bias(1, 3);

	tf2 expectForward(5, 3);
	tf2 expectBackward(5, 4);
	inputX.setValues({{0.03765099, 0.97142534, 0.8405472, 0.44674567},
					  {0.68986554, 0.80415858, 0.20979938, 0.69946131},
					  {0.24729487, 0.80640938, 0.74582434, 0.14324305},
					  {0.80681086, 0.90040793, 0.51829794, 0.77721708},
					  {0.42278581, 0.34412754, 0.78025222, 0.5804827}});

	weight.setValues({{0.09216211, 0.49851056, 0.67359579},
					  {0.21330701, 0.91796532, 0.30249352},
					  {0.26688717, 0.27553562, 0.96643816},
					  {0.17035219, 0.75909068, 0.60404623}});

	bias.setValues({{0.75297194, 0.54182269, 0.70450569}});

	expectForward.setValues({{1.26408913, 2.02304805, 2.10590903},
							 {1.16323159, 2.2126794, 2.03771407},
							 {1.17122864, 1.71959327, 1.92233461},
							 {1.29012035, 2.50335746, 2.4907156},
							 {1.17246739, 1.72410931, 2.19809269}});

	expectBackward.setValues({{2.54354339, 2.76375086, 2.9300218, 3.02308367},
							  {2.58284554, 2.89568372, 2.8894482, 3.10865685},
							  {2.26005481, 2.40985204, 2.64421261, 2.66602756},
							  {3.04458588, 3.32661238, 3.44120331, 3.6245575},
							  {2.44816974, 2.49767687, 2.91229067, 2.83623729}});

	TensorsWithNames<tf2> argWB{{"weight", weight}, {"bias", bias}};
	AffineLayer<tf2, Device::NON_OPTIMIZE> al{argWB};
	tf2 outputX = al.forward(TensorsWithNames<tf2>{{"x", inputX}});
	bool isSame = tensorIsApprox(outputX, expectForward);

	BOOST_TEST(isSame);
	al.backward(outputX);
	auto backwardResult = al.backward(outputX);
	isSame = tensorIsApprox(expectBackward, backwardResult.at("dx"));
	BOOST_TEST(isSame);
}