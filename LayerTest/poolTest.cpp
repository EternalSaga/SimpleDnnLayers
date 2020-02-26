#include <iostream>
#include <cassert>
#include <Layers/ActivationLayers.hpp>
#include "TestUtils.hpp"
using namespace RLDNN;
using namespace RLDNN::TEST;

int main()
{
	Tensor<float, 4, RowMajor> input(3, 3, 7, 11);
	Tensor<float, 2, RowMajor> kernel(2, 2);
	Tensor<float, 4, RowMajor> output(3, 2, 6, 11);
	input.setRandom();
	kernel.setRandom();

	Eigen::array<int, 2> dims({ 1, 2 });  // Specify second and third dimension for convolution.

	output = input.convolve(kernel, dims);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 6; ++k) {
				for (int l = 0; l < 11; ++l) {
					const float result = output(i, j, k, l);
					const float expected = input(i, j + 0, k + 0, l) * kernel(0, 0) +
						input(i, j + 1, k + 0, l) * kernel(1, 0) +
						input(i, j + 0, k + 1, l) * kernel(0, 1) +
						input(i, j + 1, k + 1, l) * kernel(1, 1);
					assert(result == expected);
				}
			}
		}
	}
	std::cout<<"bbb"<<std::endl;
    return 0;
}

