#include "OpInterfaces.hpp"
namespace RLDNN
{
template <typename Precision, size_t Rank>
class RelULayer: public LayerInterface<RelULayer<Precision,Rank>,Precision,Rank>
{
private:
    Tensor<bool, Rank> mask;
public:
    RelULayer() = default;
    ~RelULayer() = default;
    Tensor<Precision, Rank> forwardImpl(const TensorsWithNames<Precision, Rank> &args)
    {
        const Tensor<Precision, Rank> &x = args.at("x");
        this->mask = x > 0.f;
        Tensor<Precision, Rank> zeros(x.dimensions());
        zeros.setZero();
        Tensor<Precision, Rank> out = mask.select(x, zeros);
        return out;
    }
    TensorsWithNames<Precision, Rank> backward(const Tensor<Precision, Rank> &dout)
    {
        TensorsWithNames<Precision, Rank> gradient;
        gradient["dx"] = mask.template cast<Precision>();
        return gradient;
    }
};


template <typename Precision, size_t Rank>
class SigmoidLayer: public LayerInterface<SigmoidLayer<Precision,Rank>,Precision,Rank>{
private:
    Tensor<Precision, Rank> out;
public:
    SigmoidLayer()=default;
    ~SigmoidLayer()=default;
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
