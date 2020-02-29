#pragma once
#include "OpInterfaces.hpp"

extern "C"
{
#include "Non_Opt_Affine_C.h"
}

namespace RLDNN
{
template <typename TensorType, Device dev>
class AffineLayer : public LayerInterface<AffineLayer<TensorType, dev>, TensorType, dev>
{
private:
	TensorType weight;
	TensorType bias;
	TensorType x;
	TensorType dW;
	Eigen::Tensor<typename TensorType::Scalar, 1> dB;
	constexpr static size_t lastIdx = (TensorType::NumDimensions - 1);

public:
	AffineLayer() = delete;
	AffineLayer(TensorsWithNames<TensorType> weightBias) : weight(weightBias.at("weight")),
														   bias(weightBias.at("bias")) {}
	~AffineLayer() = default;
	TensorType forwardImpl(const TensorsWithNames<TensorType> &argX)
	{
		this->x = argX.at("x");
		if constexpr (dev == Device::CPU)
		{
			Eigen::array<Eigen::IndexPair<int>, 1> productDims = {Eigen::IndexPair<int>(lastIdx, lastIdx - 1)};
			TensorType out = x.contract(weight, productDims) + bias.broadcast(Eigen::array<int, TensorType::NumDimensions>({static_cast<int>(x.dimension(0)), 1}));
			return out;
		}
		else if (dev == Device::NON_OPTIMIZE)
		{
			TensorType out(x.dimension(0), weight.dimension(1));
			out.setZero();
			ForwardArgs fa{x.data(),weight.data(),bias.data(),x.dimension(0),x.dimension(1),weight.dimension(1)};
			affineForward(fa, out.data(), out.size());
			return out;
		}
	}
	TensorsWithNames<TensorType> backwardImpl(
		const TensorType &inputD)
	{
		if constexpr (dev == Device::CPU)
		{
			//Make matrix production on the last two dimension，transpose weight
			Eigen::array<Eigen::IndexPair<int>, 1> productDims = {Eigen::IndexPair<int>(lastIdx, lastIdx)};
			TensorType dx(inputD.contract(this->weight, productDims));
			//Make matrix production on the last two dimension, transpose x
			productDims = {Eigen::IndexPair<int>{lastIdx - 1, lastIdx - 1}};
			this->dW = this->x.contract(inputD, productDims);
			//Reduce by dimension N
			this->dB = inputD.sum(Eigen::array<uint32_t, 1>({0}));
			return TensorsWithNames<TensorType>{{"dx", dx}};
		}
		else if (dev == Device::NON_OPTIMIZE)
		{
			TensorType dx(x.dimension(0),weight.dimension(0));//C
			this->dW = TensorType(this->x.dimension(1),this->weight.dimension(1));
			this->dB = Eigen::Tensor<typename TensorType::Scalar, 1>(this->weight.dimension(1));
			dx.setZero();dW.setZero();dB.setZero();
			BackwardArgs ba{inputD.data(),weight.data(),bias.data(),{inputD.dimension(0),inputD.dimension(1)},{weight.dimension(0),weight.dimension(1)},bias.dimension(0)};
			BackwardOut bo{dx.data(),dW.data(),dB.data(),{dx.dimension(0),dx.dimension(1)}};
			affineBackward(ba,bo);
			return TensorsWithNames<TensorType>{{"dx", dx}};
		}
	}
};
} // namespace RLDNN