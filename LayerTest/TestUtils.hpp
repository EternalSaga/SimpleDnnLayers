#pragma once
#include "Layers/OpInterfaces.hpp"
namespace RLDNN {
	namespace TEST {
        template <typename TensorType>
        bool tensorIsApprox(const TensorType& a, const TensorType& b)
        {
            //without const_cast, why it can be compiled by g++?
            auto unconstAdata = const_cast<typename TensorType::Scalar*>(a.data());
            auto unconstBdata = const_cast<typename TensorType::Scalar*>(b.data());
            Eigen::Map<Eigen::VectorX<typename TensorType::Scalar>> ma(unconstAdata, a.size());
            Eigen::Map<Eigen::VectorX<typename TensorType::Scalar>> mb(unconstBdata, b.size());
            return ma.isApprox(mb);
        }
	}
}