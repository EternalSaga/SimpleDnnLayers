#include "Non_Opt_Convolution_C.h"
void for_dot(float* conv,const float* patches,const FiveDimShape patches_Shape,const float* filter,const FourDimShape filter_Shape)
{
    for(int n=0;n<patches_Shape.dim0BeforeTrans;n++){
        for(int d=0;d<filter_Shape.dim0BeforeTrans;d++){ 
            for(int k=0;k<patches_Shape.dim1BeforeTrans;k++){
                float sum=0;
                for(int c=0;c<patches_Shape.dim4BeforeTrans;c++){
                    for(int h=0;h<patches_Shape.dim2BeforeTrans;h++){
                        for(int w=0;w<patches_Shape.dim3BeforeTrans;w++){
                            //sum+=patches(n,k,h,w,c)*m_filter(d,h,w,c);
                            sum+=patches[(((n*patches_Shape.dim1BeforeTrans+k)*patches_Shape.dim2BeforeTrans+h)*patches_Shape.dim3BeforeTrans+w)*patches_Shape.dim4BeforeTrans+c]
                                    *filter[((d*patches_Shape.dim2BeforeTrans+h)*patches_Shape.dim3BeforeTrans+w)*patches_Shape.dim4BeforeTrans+c];
                        }  
                    }
                }
                //(n,k,d)=sum;
                conv[(n*patches_Shape.dim1BeforeTrans+k)*filter_Shape.dim0BeforeTrans+d]=sum;
            }
        }
    }
}
void db_dot(float* dB,const float* dout,const FourDimShape dout_Shape)
{
    for(int d=0;d<dout_Shape.dim3BeforeTrans;d++){
        for(int n=0;n<dout_Shape.dim0BeforeTrans;n++){
            for(int i=0;i<dout_Shape.dim1BeforeTrans;i++){
                for(int j=0;j<dout_Shape.dim2BeforeTrans;j++){
                    dB[d]+=dout[((n*dout_Shape.dim1BeforeTrans+i)*dout_Shape.dim2BeforeTrans+j)*dout_Shape.dim3BeforeTrans+d];
                }
            }
        }
    }
}
void dw_dot(float* dW,const FourDimShape dW_Shape,const float* dout,const FourDimShape dout_Shape,const float* x,const FourDimShape x_Shape)
{
    for(int d=0;d<dW_Shape.dim0BeforeTrans;d++){
        for(int h=0;h<dW_Shape.dim1BeforeTrans;h++){
            for(int w=0;w<dW_Shape.dim2BeforeTrans;w++){
                for(int c=0;c<dW_Shape.dim3BeforeTrans;c++){
                    for(int n=0;n<dout_Shape.dim0BeforeTrans;n++){
                        for(int i=0;i<dout_Shape.dim1BeforeTrans;i++){
                            for(int j=0;j<dout_Shape.dim2BeforeTrans;j++){
                                //dW(d,h,w,c)+=dout(n,i,j,d)*x(n,i+h,j+w,c);
                                dW[(((d*dW_Shape.dim1BeforeTrans)+h)*dW_Shape.dim2BeforeTrans+w)*dW_Shape.dim3BeforeTrans+c]
                                        +=dout[((n*dout_Shape.dim1BeforeTrans+i)*dout_Shape.dim2BeforeTrans+j)*dW_Shape.dim0BeforeTrans+d]
                                        *x[((n*x_Shape.dim1BeforeTrans+i+h)*x_Shape.dim2BeforeTrans+j+w)*dW_Shape.dim3BeforeTrans+c];

                            }
                        }
                    }
                }
            }
        }
    }
}
void filter_dot(float* flip_filter,const FourDimShape flip_filter_Shape,const float* filter,const FourDimShape filter_Shape)
{
    for(int d=0;d<filter_Shape.dim0BeforeTrans;d++){
        for(int h=0;h<filter_Shape.dim1BeforeTrans;h++){
            for(int w=0;w<filter_Shape.dim2BeforeTrans;w++){
                for(int c=0;c<filter_Shape.dim3BeforeTrans;c++){
                    //flip_filter(m_filter.dimension(0)-d-1,m_filter.dimension(1)-h-1,m_filter.dimension(2)-w-1,m_filter.dimension(3)-c-1)=m_filter(d,h,w,c);
                    flip_filter[(((filter_Shape.dim0BeforeTrans-d-1)*flip_filter_Shape.dim1BeforeTrans+filter_Shape.dim1BeforeTrans-h-1)*flip_filter_Shape.dim2BeforeTrans+filter_Shape.dim2BeforeTrans-w-1)*flip_filter_Shape.dim3BeforeTrans+filter_Shape.dim3BeforeTrans-c-1]
                            =filter[((d*filter_Shape.dim1BeforeTrans+h)*filter_Shape.dim2BeforeTrans+w)*filter_Shape.dim3BeforeTrans+c];
                }
            }
        }
    }
}
void bac_dot(float* conv,const float* patches,const FiveDimShape patches_Shape,const float* flip_filter,const FourDimShape flip_filter_Shape)
{
    for(int n=0;n<patches_Shape.dim0BeforeTrans;n++){
        for(int d=0;d<flip_filter_Shape.dim0BeforeTrans;d++){ 
            for(int k=0;k<patches_Shape.dim1BeforeTrans;k++){
                float sum=0;
                for(int c=0;c<patches_Shape.dim4BeforeTrans;c++){
                    for(int h=0;h<patches_Shape.dim2BeforeTrans;h++){
                        for(int w=0;w<patches_Shape.dim3BeforeTrans;w++){
                            //sum+=patches(n,k,h,w,c)*flip_filter(d,h,w,c);
                            sum+=patches[(((n*patches_Shape.dim1BeforeTrans+k)*patches_Shape.dim2BeforeTrans+h)*patches_Shape.dim3BeforeTrans+w)*patches_Shape.dim4BeforeTrans+c]
                                    *flip_filter[((d*patches_Shape.dim2BeforeTrans+h)*patches_Shape.dim3BeforeTrans+w)*patches_Shape.dim4BeforeTrans+c];
                        }  
                    }
                }
                conv[(n*patches_Shape.dim1BeforeTrans+k)*flip_filter_Shape.dim0BeforeTrans+d]=sum;
            }
        }
    }/**/
}
void padding_dot(float* padding_x,const FourDimShape padding_x_Shape,const TwoDimShape in_Shape,const float* argX,const FourDimShape argX_Shape)
{
    for(int n=0;n<argX_Shape.dim0BeforeTrans;n++){
        for(int d=0;d<argX_Shape.dim3BeforeTrans;d++){
            for(int i=0;i<argX_Shape.dim1BeforeTrans;i++){
                for(int j=0;j<argX_Shape.dim2BeforeTrans;j++){
                    padding_x[((n*padding_x_Shape.dim1BeforeTrans+i+(in_Shape.dim0BeforeTrans-argX_Shape.dim1BeforeTrans)/2)*padding_x_Shape.dim2BeforeTrans+j+(in_Shape.dim1BeforeTrans-argX_Shape.dim2BeforeTrans)/2)*padding_x_Shape.dim3BeforeTrans+d]
                            =argX[((n*argX_Shape.dim1BeforeTrans+i)*argX_Shape.dim2BeforeTrans+j)*argX_Shape.dim3BeforeTrans+d];
                }
            }
        }
    }
}
void convert(float* out,const FourDimShape out_Shape,const float* x,const FourDimShape x_Shape)
{
    for(int n=0;n<x_Shape.dim0BeforeTrans;n++){
        for(int h=0;h<x_Shape.dim1BeforeTrans;h++){
            for(int w=0;w<x_Shape.dim2BeforeTrans;w++){
                for(int c=0;c<x_Shape.dim3BeforeTrans;c++){
                    out[((n*x_Shape.dim3BeforeTrans+c)*x_Shape.dim1BeforeTrans+h)*x_Shape.dim2BeforeTrans+w]
                            =x[((n*x_Shape.dim1BeforeTrans+h)*x_Shape.dim2BeforeTrans+w)*x_Shape.dim3BeforeTrans+c];
                }
            }
        }
    }
}
