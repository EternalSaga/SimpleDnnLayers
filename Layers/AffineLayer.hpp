#pragma once

#include "OpInterfaces.hpp"

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
		TensorType forwardImpl(const TensorsWithNames<TensorType>& argX) {
			this->x = argX.at("x");
			Eigen::array<Eigen::IndexPair<int>, 1> productDims = { Eigen::IndexPair<int>(lastIdx,lastIdx - 1) };
			TensorType out = x.contract(weight, productDims) + bias;
			return out;
		}
		TensorsWithNames<TensorType> backward(
			const TensorType& inputD) {
			//Make matrix production on the last two dimension，transpose weight
			Eigen::array<Eigen::IndexPair<int>, 1> productDims = { Eigen::IndexPair<int>(lastIdx, lastIdx) };
			TensorType dx(inputD.contract(this->weight, productDims));
			//Make matrix production on the last two dimension, transpose x
			productDims = { Eigen::IndexPair<int>{lastIdx-1, lastIdx - 1} };
			this->dW = this->x.contract(inputD, productDims);
			//Reduce by dimension N
			this->dB = inputD.sum(Eigen::array<uint32_t, 1>({ 0 }));
			return TensorsWithNames<TensorType>{ {"dx", dx}};
		}
	};
}