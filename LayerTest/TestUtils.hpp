#pragma once
#include "Layers/OpInterfaces.hpp"
namespace RLDNN {
	namespace TEST {
        template <typename Precision, size_t Rank>
        bool tensorIsApprox(const Eigen::Tensor<Precision, Rank>& a, const Eigen::Tensor<Precision, Rank>& b)
        {
            //without const_cast, why it can be compiled by g++?
            auto unconstAdata = const_cast<Precision*>(a.data());
            auto unconstBdata = const_cast<Precision*>(b.data());
            Eigen::Map<Eigen::VectorX<Precision>> ma(unconstAdata, a.size());
            Eigen::Map<Eigen::VectorX<Precision>> mb(unconstBdata, b.size());
            return ma.isApprox(mb);
        }
	}
}