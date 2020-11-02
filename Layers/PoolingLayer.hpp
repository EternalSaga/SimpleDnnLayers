#pragma once
#include "OpInterfaces.hpp"



#include <math.h>
#include <vector>
#include "Non_Opt_Pooling_C.h"
enum PoolingMethod
{
    max,
    avg
};


namespace RLDNN
{
    template <typename TensorType, Device dev>
    class PoolingLayer 
    {
    private:
        int m_hksize;
        int m_wksize;
        int m_hstride;
        int m_wstride;

        PaddingType m_padding_type;
        PoolingMethod m_pooling_method;

        Tensor4xf x;
       
        Eigen::Tensor<Mask, 4,Eigen::RowMajor> maxmask;
       
    public:
        PoolingLayer() = delete;
        PoolingLayer( std::vector<int> pool_size,std::vector<int> strides, PoolingMethod pooling_method,PaddingType padding_type = PADDING_VALID)
        {
            m_hksize = pool_size[0];
            m_wksize = pool_size[1];
            m_hstride = strides[0];
            m_wstride = strides[1];

            m_padding_type = padding_type;
            m_pooling_method = pooling_method;
            
        }
        ~PoolingLayer() = default;


      
        TensorType forwardImpl(TensorsWithNames<TensorType> argX,std::ofstream& outt)
        {
            this->x = argX.at("x");
            Eigen::array<int, 2> reduction_dims{ 2,3 };

            Eigen::DSizes<int, 4> post_reduce_dims = get_top_shape(x);
            Tensor5xf patches = x.extract_image_patches(m_hksize, m_wksize, m_hstride, m_wstride, 1, 1, Eigen::PADDING_VALID);
            if(m_padding_type==PADDING_SAME)
            {
                post_reduce_dims = get_top_shape_same(x);
                patches = x.extract_image_patches(m_hksize, m_wksize, m_hstride, m_wstride, 1, 1, Eigen::PADDING_SAME);
            }
            
            Tensor3xf pooling(post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]);
            Eigen::Tensor<Mask, 3,Eigen::RowMajor> mask(post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]);
      
            
            TensorType out;
            if constexpr (dev == Device::CPU)
            {
                switch (m_pooling_method)
                {
                case avg:
          
                    pooling= patches.mean(reduction_dims);
                    writerT(pooling, outt);
                   
                    out = pooling.reshape(post_reduce_dims);
                   
                    break;

                case max:
                    pooling=maximum_cc(patches,reduction_dims, post_reduce_dims, mask);
                    writerT(pooling, outt);
                   
                    out = pooling.reshape(post_reduce_dims);
                   
                    maxmask = mask.reshape(post_reduce_dims);
                   
                    break;
                default:
                    break;/**/
                }
            }
            else if (dev == Device::NON_OPTIMIZE)
            {
                switch (m_pooling_method)
                {
                case avg:

                    mean_c(pooling.data(),patches.data(),{2,3}, {post_reduce_dims[0], post_reduce_dims[1] , post_reduce_dims[2], post_reduce_dims[3]},{patches.dimension(0),patches.dimension(1),patches.dimension(2),patches.dimension(3),patches.dimension(4)});
              
                    out = pooling.reshape(post_reduce_dims);
                  
                    break;

                case max:
                    maximum_c(pooling.data(),patches.data(),{2,3}, {post_reduce_dims[0], post_reduce_dims[1] , post_reduce_dims[2], post_reduce_dims[3]},{patches.dimension(0),patches.dimension(1),patches.dimension(2),patches.dimension(3),patches.dimension(4)},mask.data());
                   
                    out = pooling.reshape(post_reduce_dims);
                    
                    
                    maxmask = mask.reshape(post_reduce_dims);
                    break;
                default:
                    break;
                }
            }
           
            return out;

        }  
        TensorsWithNames<TensorType> backwardImpl(const Eigen::DSizes<int, 4> &orig_input_shape,Tensor4xf& dout, std::ofstream& outt)
        {
            TensorsWithNames<TensorType> argDX;

            if constexpr (dev == Device::CPU)
            {
                switch (m_pooling_method)
                {
                case avg:
                    argDX=avgpooling_backward(orig_input_shape,dout,  outt);

                    break;

                case max:
                    argDX=maxpooling_backward(orig_input_shape,dout,  outt);
                    break;
                default:
                    break;
                }
            }
            else if (dev == Device::NON_OPTIMIZE)
            {
                switch (m_pooling_method)
                {
                case avg:
                    argDX=avgpooling_backward(orig_input_shape,dout,  outt);
                    break;

                case max:
                    argDX=maxpooling_backward(orig_input_shape,dout,  outt);
                    break;
                default:
                    break;
                }
            }
            return argDX;
        }
      
    private:
        Eigen::DSizes<int, 4> get_top_shape(const Tensor4xf& bottom)
        {
            Eigen::DSizes<int, 4>top_shape;
            top_shape[0] = bottom.dimension(0);
            top_shape[1] = Eigen::divup(float(bottom.dimension(1) - m_hksize + 1), float(m_hstride));
            top_shape[2] = Eigen::divup(float(bottom.dimension(2) - m_wksize + 1), float(m_wstride));
            top_shape[3] = bottom.dimension(3);
            return top_shape;
        }
        Eigen::DSizes<int, 4> get_top_shape_same(const Tensor4xf& bottom)
        {
            Eigen::DSizes<int, 4>top_shape;
            top_shape[0] = bottom.dimension(0);
            top_shape[1] = ceil(Eigen::divup(float(bottom.dimension(1) - m_hksize + 1), float(m_hstride)));
            top_shape[2] = ceil(Eigen::divup(float(bottom.dimension(2) - m_wksize + 1), float(m_wstride)));
            top_shape[3] = bottom.dimension(3);
            return top_shape;
        }
        
        Tensor3xf maximum_cc(Tensor5xf patches, Eigen::array<int, 2> reduction_dims, Eigen::DSizes<int, 4>post_reduce_dims, Eigen::Tensor<Mask, 3,Eigen::RowMajor>& mask) {
            Tensor3xf out(post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]);

            for (int i = 0; i < patches.dimension(0); i++)
            {
                for (int j = 0; j < patches.dimension(4); j++)
                {
                    for (int k = 0; k < patches.dimension(1 + 2 + 3 - reduction_dims[0] - reduction_dims[1]); k++)
                    {
                        int p_key = 0; int q_key = 0;
                        for (int p = 0; p < patches.dimension(reduction_dims[0]); p++)
                        {
                            for (int q = 0; q < patches.dimension(reduction_dims[1]); q++)
                            {
                                if (patches(i,k ,p, q,  j) > patches(i,k, p_key, q_key, j)) { p_key = p; q_key = q; }
                            }
                        }
                       out(i, k, j) = patches(i, k, p_key, q_key, j);
                       mask(i, k, j).x = p_key;
                       mask(i, k, j).y = q_key;
                    }  
                }
            }

            return out;
        }
        
        
   
        TensorsWithNames<TensorType> avgpooling_backward(const Eigen::DSizes<int, 4> &orig_input_shape,const Tensor4xf& dout, std::ofstream& outt)
        {
            
        
        
            Tensor4xf dX(orig_input_shape[0],orig_input_shape[1],orig_input_shape[2],orig_input_shape[3]);
            dX.setZero();

            for (int i = 0; i < dout.dimension(0); i++)
            {
                for (int j = 0; j < dout.dimension(3); j++)
                {
        
                    for (int p = 0; p < dout.dimension(1); p++)
                    {
                        for (int q = 0; q < dout.dimension(2); q++)
                        {

                            for (int k = p * m_wstride;( k < p * m_wstride + m_wksize)&&(k<dX.dimension(1)); k++)
                            {
                                for (int g = q*m_hstride; ( g < q * m_hstride + m_hksize)&&(g<dX.dimension(2)); g++)
                                {
                                    dX(i, k, g, j) += dout(i, p, q, j);//??????
                                }
                            }
                        }
                    }   
                }
            }
           
            return TensorsWithNames<TensorType>{ {"dX", dX}};
        }
        TensorsWithNames<TensorType> maxpooling_backward(const Eigen::DSizes<int, 4> &orig_input_shape,const Tensor4xf& dout, std::ofstream& outt)
        {
            
            //(,)
            Tensor4xf dX(orig_input_shape[0],orig_input_shape[1],orig_input_shape[2],orig_input_shape[3]);
            dX.setZero();
            for (int i = 0; i < dout.dimension(0); i++)
            {
                for (int j = 0; j < dout.dimension(3); j++)
                {

                     for (int p = 0; p < dout.dimension(1); p++)
                     {
                         for (int q = 0; q < dout.dimension(2); q++)
                         {
                             for (int k = p*m_wstride;( k < p * m_wstride + m_wksize)&&(k<dX.dimension(1)); k++)
                             {
                                 for (int g = q*m_hstride;( g < q * m_hstride + m_hksize)&&(g<dX.dimension(2)); g++)
                                 {
                                     int a = k - p * m_wstride; int aa = maxmask(i, p, q, j).x;
                                     int b = g - q * m_hstride; int bb = maxmask(i, p, q, j).y;
                                 
                                     if ((k- p * m_wstride )== maxmask(i, p, q, j).x &&(( g- q*m_hstride) == maxmask(i, p, q, j).y))
                                     {
                                         dX(i, k, g, j) = dout(i, p, q, j);
                                       
                                     }
                                 }
                             }
                    
                         }
                  
                     }
                     
                }
            }
        
            return TensorsWithNames<TensorType>{ {"dX", dX}};
        }

    };
}
