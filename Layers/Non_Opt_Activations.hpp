#include "OpInterfaces.hpp"
extern "C"{
#include "Non_Opt_Activations_C_impl.h"
}

namespace RLDNN
{
template <typename Precision, size_t Rank, Device dev = Device::NON_OPTIMIZE>
class RelULayer
{
    size_t size;
    Eigen::Array<Rank> dimensions;
public:
    RelULayer() = default;
    ~RelULayer() = default;
    Tensor<Precision, Rank> forward(const OpInOutType<Precision, Rank> &args)
    {
        const Tensor<Precision, Rank> &x = args.at("x");
        this->size=out.size();
        this->dimensions = x.dimensions();
        Tensor<Precision, Rank> out(dimensions);
        assert(out.size()==x.size());
        batchLeakyForward(x.data(),out.data(),size);
        return out;
    }
    OpInOutType<Precision, Rank> backward(const Tensor<Precision, Rank> &dout)
    {
        OpInOutType<Precision, Rank> gradient;
        gradient["dx"] = Tensor<Precision, Rank>(dimensions);
        batchLeakyBackward(dout.data(),gradient["dx"].data(),size);
        return gradient;
    }
};
} // namespace RLDNN
