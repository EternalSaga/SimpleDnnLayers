#pragma once
#include "OpInterfaces.hpp"

namespace RLDNN
{
	//dimension's order is assumed as NCHW
	template <typename TensorType, Device dev>
	class AffineLayer : public LayerInterface<AffineLayer<TensorType, dev>, TensorType, dev>
	{
	private:
		TensorType weight;
		TensorType bias;
		TensorType x;
		Eigen::array<uint32_t, TensorType::NumDimensions> originalXShape;
		TensorType dW;

		Eigen::Tensor<typename TensorType::Scalar,1> dB;

	public:
		AffineLayer() = delete;
		AffineLayer(TensorsWithNames<TensorType> weightBias) : weight(weightBias.at("weight")),
			bias(weightBias.at("bias")) {}
		~AffineLayer() = default;
		TensorType forwardImpl(const TensorsWithNames<TensorType>& argX) {
			this->x = argX.at("x");
			TensorType out = x * weight + bias;
			return out;
		}
		TensorsWithNames<TensorType> backward(
			const TensorType& inputD) {
			TensorType dx(inputD * (Eigen::Transpose(this->weight)));
			this->dW = this->x * inputD;
			//Reduce by dimension N
			this->dB = inputD.sum(Eigen::array<uint32_t,1>({ 0 }));
			dx.reshape(this->originalXShape);
			return TensorsWithNames<TensorType>{ {"dx", dx}};
		}
	};

} // namespace RLDNN
