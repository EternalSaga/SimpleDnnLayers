#include <stdint.h>
static inline float relu_activate(float x){return x*(x>0);}
static inline float relu_gradient(float x){return (x>0);}
static inline float leaky_activate(float x,float alpha){return (x>0) ? x : alpha *x;}
static inline float leaky_gradient(float x,float alpha){return (x>0) ? 1 : alpha;}

void batchReluForward(const float* input,float* output,const int32_t* size);
void batchReluBackward(const float* input,float* output,const int32_t* size);
