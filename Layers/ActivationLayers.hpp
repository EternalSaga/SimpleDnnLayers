#include "OpInterfaces.hpp"
extern "C"{
#include "Non_Opt_Activations_C_impl.h"
}
namespace RLDNN
{
template <typename TensorType, Device dev>
class RelULayer : public LayerInterface<RelULayer<TensorType,dev>,TensorType,dev>
{
private:
    Tensor<bool, TensorType::NumDimensions> mask;

public:
    RelULayer() = default;
    ~RelULayer() = default;
    TensorType forwardImpl(const TensorsWithNames<TensorType> &args)
    {
        const TensorType &x = args.at("x");
        if constexpr (dev == Device::CPU)
        {
            this->mask = x > 0.f;
            TensorType zeros(x.dimensions());
            zeros.setZero();
            TensorType out = mask.select(x, zeros);
            return out;
        }
        else if(dev == Device::NON_OPTIMIZE)
        {
            TensorType out(x.dimensions());
            const TensorType &x = args.at("x");
            batchReluForward(x.data(), out.data(), x.size());
            return out;
        }
        
    }
    TensorsWithNames<TensorType> backward(const TensorType &dout)
    {
        
        TensorsWithNames<TensorType> gradient;
        if constexpr (dev == Device::CPU){
            gradient["dx"] = mask.template cast<TensorType::Scalar>();
        }
        else if(dev == Device::NON_OPTIMIZE){
            gradient["dx"] = TensorType(dout.dimensions());
        batchReluBackward(dout.data(),gradient["dx"].data(),dout.size());
        }
        return gradient;
    }
};

template <typename TensorType, Device dev = Device::CPU>
class SigmoidLayer : public LayerInterface<SigmoidLayer<TensorType>,TensorType,dev>
{
private:
    TensorType out;

public:
    SigmoidLayer() = default;
    ~SigmoidLayer() = default;
    TensorType forwardImpl(const TensorsWithNames<TensorType> &args)
    {
        auto negIn = -args.at("x");
        this->out = 1.f / (1.f + negIn.exp());
        return out;
    }
    TensorsWithNames<TensorType> backward(const TensorType &dout)
    {
        TensorsWithNames<TensorType> gradient;
        gradient["dx"] = dout * (1.0 - this->out) * this->out;
        return gradient;
    }
};

} // namespace RLDNN
