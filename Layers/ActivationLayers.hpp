#include "OpInterfaces.hpp"
extern "C"{
#include "Non_Opt_Activations_C_impl.h"
}
namespace RLDNN
{
template <typename Precision, size_t Rank, Device dev = Device::CPU>
class RelULayer : public LayerInterface<RelULayer<Precision, Rank>, Precision, Rank>
{
private:
    Tensor<bool, Rank> mask;

public:
    RelULayer() = default;
    ~RelULayer() = default;
    Tensor<Precision, Rank> forwardImpl(const TensorsWithNames<Precision, Rank> &args)
    {
        if constexpr (dev == Device::CPU)
        {
            const Tensor<Precision, Rank> &x = args.at("x");
            this->mask = x > 0.f;
            Tensor<Precision, Rank> zeros(x.dimensions());
            zeros.setZero();
            Tensor<Precision, Rank> out = mask.select(x, zeros);
        }
        else if(dev == Device::NON_OPTIMIZE)
        {
            Tensor<Precision, Rank> out(dimensions);
            const Tensor<Precision, Rank> &x = args.at("x");
            this->size = out.size();
            this->dimensions = x.dimensions();

            assert(out.size() == x.size());
            batchReluForward(x.data(), out.data(), size);
        }
        return out;
    }
    TensorsWithNames<Precision, Rank> backward(const Tensor<Precision, Rank> &dout)
    {
        
        TensorsWithNames<Precision, Rank> gradient;
        if constexpr (dev == Device::CPU){
            gradient["dx"] = mask.template cast<Precision>();
        }
        else if(dev == Device::NON_OPTIMIZE){
            gradient["dx"] = Tensor<Precision, Rank>(dimensions);
        batchReluBackward(dout.data(),gradient["dx"].data(),size);
        }
        return gradient;
    }
};

template <typename Precision, size_t Rank>
class SigmoidLayer : public LayerInterface<SigmoidLayer<Precision, Rank>, Precision, Rank>
{
private:
    Tensor<Precision, Rank> out;

public:
    SigmoidLayer() = default;
    ~SigmoidLayer() = default;
    Tensor<Precision, Rank> forwardImpl(const TensorsWithNames<Precision, Rank> &args)
    {
        auto negIn = -args.at("x");
        this->out = 1.f / (1.f + negIn.exp());
        return out;
    }
    TensorsWithNames<Precision, Rank> backward(const Tensor<Precision, Rank> &dout)
    {
        TensorsWithNames<Precision, Rank> gradient;
        gradient["dx"] = dout * (1.0 - this->out) * this->out;
        return gradient;
    }
};

} // namespace RLDNN
