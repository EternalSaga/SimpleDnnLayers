#include "Non_Opt_Affine_C.h"
void affineForward(ForwardArgs fa, float* outdata)
{
    simplifiedSgemm(CblasRowMajor,fa.x,CblasNoTrans,fa.weight,CblasNoTrans,outdata,fa.m,fa.n,fa.k);
    add_bias(outdata,fa.bias,fa.m,fa.n,1);
}

void affineBackward(BackwardArgs ba, BackwardOut bo){

}