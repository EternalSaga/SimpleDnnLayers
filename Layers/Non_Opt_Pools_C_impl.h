#include <stdint.h>
//filterSize、stride

void resizeMaxPoolLayer();
void maxPoolForward(const float* input,float* output,const int32_t size);
void maxPoolBackward(const float* input,float* output,const int32_t size);

void resizeAvgPoolLayer();
void avgPoolForward(const float* input,float* output,const int32_t size);
void avgPoolBackward(const float* input,float* output,const int32_t size);
