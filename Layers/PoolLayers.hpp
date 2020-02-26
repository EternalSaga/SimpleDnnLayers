#include "OpInterfaces.hpp"
extern "C"{
#include "Non_Opt_Activations_C_impl.h"
}
namespace RLDNN
{
    template <typename TensorType, Device dev = Device::CPU>
    class MaxPoolLayer : public LayerInterface<MaxPoolLayer<TensorType,dev>,TensorType,dev>
    {
    private:
        Tensor<double, TensorType::NumDimensions> mask;

    public:
        MaxPoolLayer() = default;
        ~MaxPoolLayer() = default;
        TensorType forwardImpl(const TensorsWithNames<TensorType> &args)
        {
            const TensorType &x = args.at("x");
            if constexpr (dev == Device::CPU)
            {
                TensorType out = mask.select(x, zeros);
                
                return out;
            }
            else if(dev == Device::NON_OPTIMIZE)
            {
                TensorType out(x.dimensions());

                return out;
            }
            
        }
        TensorsWithNames<TensorType> backward(const TensorType &dout)
        {
            
            TensorsWithNames<TensorType> gradient;
            if constexpr (dev == Device::CPU){
                gradient["dx"] = mask.template cast<typename TensorType::Scalar>();
            }
            else if(dev == Device::NON_OPTIMIZE){
                gradient["dx"] = TensorType(dout.dimensions());
            batchReluBackward(dout.data(),gradient["dx"].data(),dout.size());
            }
            return gradient;
        }
    };
} // namespace RLDNN
