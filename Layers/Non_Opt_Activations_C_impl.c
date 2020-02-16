#include "Non_Opt_Activations_C_impl.h"

void batchReluForward(const float* input,float* output,const int32_t size){
    for (int32_t i = 0; i < size; i++)
    {
        output[i]=relu_activate(*input);
    }
}
void batchReluBackward(const float* input,float* output,const int32_t size){
    for (int32_t i = 0; i < size; i++)
    {
        output[i]=relu_gradient(*input);
    }
}