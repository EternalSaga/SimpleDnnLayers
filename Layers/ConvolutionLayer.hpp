#pragma once

#include "OpInterfaces.hpp"

#include "Non_Opt_Convolution_C.h"
#include <math.h>


//NHWC

namespace RLDNN
{
    template <typename TensorType, Device dev>
    class ConvolutionLayer 
    {
    private:
        
        Tensor4xf m_filter;
        std::vector<int> m_strides;

        PaddingType m_padding_type;
       
	    Tensor4xf x;
	    Tensor4xf dW;
	    Tensor1xf dB;
        
    public:
        ConvolutionLayer() = delete;
        ConvolutionLayer(Tensor4xf filter,std::vector<int> strides,PaddingType padding_type = PADDING_VALID)
        {
            
            m_strides = strides;
            m_filter=filter;

            m_padding_type = padding_type;
            
            
        }
        ~ConvolutionLayer() = default;
        TensorType forwardImpl(TensorsWithNames<TensorType> argX){
            Tensor4xf out;
            x = argX.at("x");
            if(m_padding_type==PADDING_SAME)x=x_padding(x,x.dimension(1),x.dimension(2),m_strides);
            //row ( ,*,w,h, )
            Tensor5xf patches=x.extract_image_patches(m_filter.dimension(2), m_filter.dimension(1), m_strides[1], m_strides[0], 1, 1, Eigen::PADDING_VALID);
            Tensor3xf conv(patches.dimension(0), patches.dimension(1), m_filter.dimension(0));
            
            for(int n=0;n<patches.dimension(0);n++){
                 for(int d=0;d<m_filter.dimension(0);d++){ 
                    for(int k=0;k<patches.dimension(1);k++){
                        float sum=0;
                        for(int c=0;c<patches.dimension(4);c++){
                            for(int h=0;h<patches.dimension(2);h++){
                                for(int w=0;w<patches.dimension(3);w++){
                                    sum+=patches(n,k,h,w,c)*m_filter(d,h,w,c);
                                }  
                            }
                        }
                        conv(n,k,d)=sum;
                    }
                }

            }
            Eigen::array<int, 2> out_=nopadding_out_h_w();
            out=conv.reshape(Eigen::DSizes<int, 4> {patches.dimension(0), out_[0],out_[1], m_filter.dimension(0)});
            return out;
        }
        
        TensorsWithNames<TensorType> backwardImpl(Tensor4xf& dout){
            dB=Tensor1xf(dout.dimension(3));
            dB.setZero();
            for(int d=0;d<dout.dimension(3);d++){
                for(int n=0;n<dout.dimension(0);n++){
                    for(int i=0;i<dout.dimension(1);i++){
                        for(int j=0;j<dout.dimension(2);j++){
                            dB(d)+=dout(n,i,j,d);
                        }
                    }
                }
            }
            dW=Tensor4xf (m_filter.dimension(0),m_filter.dimension(1),m_filter.dimension(2),m_filter.dimension(3));
            dW.setZero();
            for(int d=0;d<dW.dimension(0);d++){
                for(int h=0;h<dW.dimension(1);h++){
                    for(int w=0;w<dW.dimension(2);w++){
                        for(int c=0;c<dW.dimension(3);c++){
                            for(int n=0;n<dout.dimension(0);n++){
                                for(int i=0;i<dout.dimension(1);i++){
                                    for(int j=0;j<dout.dimension(2);j++){
                                        dW(d,h,w,c)+=dout(n,i,j,d)*x(n,i+h,j+w,c);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //outt<<"dout:";
           // writerT(dout,outt);
            Tensor4xf dout_padding=x_padding(dout,x.dimension(1),x.dimension(2),std::vector<int> {1,1});
            //writerT(dout_padding,outt);
            Tensor4xf flip_filter(m_filter.dimension(3),m_filter.dimension(2),m_filter.dimension(1),m_filter.dimension(0));
            for(int d=0;d<m_filter.dimension(0);d++){
                for(int h=0;h<m_filter.dimension(1);h++){
                    for(int w=0;w<m_filter.dimension(2);w++){
                        for(int c=0;c<m_filter.dimension(3);c++){
                            flip_filter(m_filter.dimension(0)-d-1,m_filter.dimension(1)-h-1,m_filter.dimension(2)-w-1,m_filter.dimension(3)-c-1)=m_filter(d,h,w,c);
                            //outt<<m_filter.dimension(1)-h-1<<":"<<h<<",";
                            //outt<<m_filter.dimension(2)-w-1<<":"<<w<<std::endl;
                        }
                    }
                }
            }
            //writerT(flip_filter,outt);
            Tensor5xf patches=dout_padding.extract_image_patches(flip_filter.dimension(2), flip_filter.dimension(1), 1, 1, 1, 1, Eigen::PADDING_VALID);
            Tensor3xf conv(patches.dimension(0), patches.dimension(1), flip_filter.dimension(0));
                
            for(int n=0;n<patches.dimension(0);n++){
                for(int d=0;d<flip_filter.dimension(0);d++){ 
                    for(int k=0;k<patches.dimension(1);k++){
                        float sum=0;
                        for(int c=0;c<patches.dimension(4);c++){
                            for(int h=0;h<patches.dimension(2);h++){
                                for(int w=0;w<patches.dimension(3);w++){
                                    sum+=patches(n,k,h,w,c)*flip_filter(d,h,w,c);
                                }  
                            }
                        }
                        conv(n,k,d)=sum;
                    }
                }
            }/**/
            Tensor4xf dX;
            //outt<<conv.dimension(0)<<","<<conv.dimension(1)<<","<<conv.dimension(2)<<std::endl;
            //outt<<x.dimension(0)<<","<<x.dimension(1)<<","<<x.dimension(2)<<","<<x.dimension(3)<<std::endl;
            dX=conv.reshape(Eigen::DSizes<int, 4> {x.dimension(0),x.dimension(1),x.dimension(2),x.dimension(3)});
            return TensorsWithNames<TensorType>{ {"dX", dX}};
            
        }
        //NHWC->NCHW
        Eigen::Tensor<float, 4,Eigen::RowMajor> convert_tensor_1 (Eigen::Tensor<float, 4,Eigen::RowMajor> x)
        {
            Eigen::Tensor<float, 4,Eigen::RowMajor> out(x.dimension(0),x.dimension(3),x.dimension(1),x.dimension(2));
            for(int n=0;n<x.dimension(0);n++){
                for(int h=0;h<x.dimension(1);h++){
                    for(int w=0;w<x.dimension(2);w++){
                        for(int c=0;c<x.dimension(3);c++){
                            out(n,c,h,w)=x(n,h,w,c);
                        }
                    }
                }
            }
            return out;
        }
        Tensor4xf get_dW()
        {
            return dW;
        }
      

    private:
        Eigen::array<int, 2> nopadding_out_h_w(){
            int out_h = 1 + int((x.dimension(1)  - m_filter.dimension(1)) / m_strides[0]);
            int out_w = 1 + int((x.dimension(2)  - m_filter.dimension(2)) / m_strides[1]);
            return Eigen::array<int, 2> {out_h,out_w};
        }
        Eigen::array<int, 2> in_h_w(int out_h,int out_w,std::vector<int> strides){
            int in_h=ceil((out_h-1)*strides[0]+m_filter.dimension(1));
            int in_w=ceil((out_w-1)*strides[1]+m_filter.dimension(2));
    
            return Eigen::array<int, 2> {in_h,in_w};
        }
        Tensor4xf x_padding(Tensor4xf &argX,int out_h,int out_w,std::vector<int> strides){
            Eigen::array<int, 2> padding =in_h_w(out_h,out_w,strides);
            
            Tensor4xf padding_x(argX.dimension(0),padding[0],padding[1],argX.dimension(3));
            padding_x.setZero();
            for(int n=0;n<argX.dimension(0);n++){
                for(int d=0;d<argX.dimension(3);d++){
                    for(int i=0;i<argX.dimension(1);i++){
                        for(int j=0;j<argX.dimension(2);j++){
                            padding_x(n,i+(padding[0]-argX.dimension(1))/2,j+(padding[1]-argX.dimension(2))/2,d)=argX(n,i,j,d);
                        }
                    }
                }
            }
            return padding_x;
        }  
    };
}