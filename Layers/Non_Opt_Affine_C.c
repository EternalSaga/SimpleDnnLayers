#include "Non_Opt_Affine_C.h"
void affineForward(ForwardArgs fa, float* outdata)
{
    simplifiedSgemm(CblasRowMajor,fa.x,CblasNoTrans,fa.weight,CblasNoTrans,outdata,fa.m,fa.n,fa.k);
    add_bias(outdata,fa.bias,fa.m,fa.n,1);
}

void affineBackward(BackwardArgs ba, BackwardOut bo){
    //Make matrix production on inputD and weight，transpose weight
    simplifiedSgemm(CblasRowMajor,ba.inputD,CblasNoTrans,ba.weight,CblasTrans,bo.dx,ba.inputDShape.dim0BeforeTrans,ba.weightShape.dim0BeforeTrans,ba.inputDShape.dim1BeforeTrans);

    //make matrix production on x and inputD, transpose x
    //simplifiedSgemm(CblasRowMajor, )
}