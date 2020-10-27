#pragma once


#include "Non_Opt_BatchNormalization_C.h"
namespace RLDNN
{
    template <typename TensorType, Device dev>
    class BatchNormalizationLayer 
    {
    private:
        Tensor1xf gamma,beta; 
        Tensor1xf dgamma,dbeta;

        float decay;
        float variance_epsilon;

        TensorType x;

        Tensor1xf batch_Mean;
        Tensor1xf batch_Var;
        int batch_size;

        Tensor2xf x_c,x_n;
        Tensor1xf std;
      
    public:
        BatchNormalizationLayer() = delete;
        BatchNormalizationLayer(Tensor1xf gamma,Tensor1xf beta,Tensor1xf batch_Mean,Tensor1xf batch_Var,float variance_epsilon,float decay=0.9)
        {
            this->gamma=gamma;
            this->beta=beta;
            this->decay=decay;
            this->variance_epsilon=variance_epsilon;
            this->batch_Mean=batch_Mean;
            this->batch_Var=batch_Var;
        }
        ~BatchNormalizationLayer() = default;
        TensorType forwardImpl(const TensorsWithNames<TensorType> &argX){
            this->x=argX.at("x");
            Tensor2xf x_2;
            if(x.dimensions().size()!=2)
            {
                x_2=x.reshape(Eigen::array<int, 2> {x.dimension(0),x.dimension(1)*x.dimension(2)*x.dimension(3)});
            }else
            {
                x_2=x;
            }
            
            if (batch_Mean.dimensions().size()==0)
            {
                batch_Mean=Tensor1xf(x_2.dimension(0));
                batch_Mean.setZero();
                batch_Var=Tensor1xf(x_2.dimension(0));
                batch_Var.setZero();
            }
            
            
            if constexpr (dev == Device::CPU)
            {
                Tensor1xf m_u=x_2.mean(Eigen::array<int, 1> {1});
                x_c=x_2-one_to_two(m_u,x_2);
                Tensor1xf var=(x_c*x_c).mean(Eigen::array<int, 1> {1});
                std=(var+zero_to_one(variance_epsilon,var)).sqrt();
                x_n=x_c/one_to_two(std,x_2);
                batch_size=x_2.dimension(0);
                batch_Mean=zero_to_one(decay,batch_Mean)*batch_Mean+zero_to_one(1-decay,batch_Mean)*m_u;
                batch_Var=zero_to_one(decay,batch_Var)*batch_Var+zero_to_one(1-decay,batch_Var)*var;
            
                x_c=x_2-one_to_two(batch_Mean,x_2);
                x_n=x_c/((one_to_two(batch_Var,x_2)+zero_to_two(variance_epsilon,x_2)).sqrt()); 
                return (one_to_two(gamma,x_2)*x_n+one_to_two(beta,x_2)).reshape(x.dimensions());

            }
            else if (dev == Device::NON_OPTIMIZE)
            { 
                Tensor1xf m_u=mean(Eigen::array<int, 1> {1},x_2);
                x_c=sub(x_2,one_to_two(m_u,x_2));
                Tensor1xf var=mean(Eigen::array<int, 1> {1},mul(x_c,x_c));
                std=sqrt_(add(var,zero_to_one(variance_epsilon,var)));
                x_n=div(x_c,one_to_two(std,x_2));
                batch_size=x_2.dimension(0);
                batch_Mean=add(mul(zero_to_one(decay,batch_Mean),batch_Mean),mul(zero_to_one(1-decay,batch_Mean),m_u));
                batch_Var=add(mul(zero_to_one(decay,batch_Var),batch_Var),mul(zero_to_one(1-decay,batch_Var),var));
            
                x_c=sub(x_2,one_to_two(batch_Mean,x_2));
                x_n=div(x_c,sqrt_(add(one_to_two(batch_Var,x_2),zero_to_two(variance_epsilon,x_2))));/**/ 
                
            }
            return (add(mul(one_to_two(gamma,x_2),x_n),one_to_two(beta,x_2))).reshape(x.dimensions());
        }

        TensorsWithNames<TensorType> backwardImpl(TensorType& dout)
        {
           
            Tensor2xf dout_2;Tensor2xf dx_c;Tensor1xf dm_u;
            if(dout.dimensions().size()!=2)
            {
                
                dout_2=dout.reshape(Eigen::array<int, 2> {dout.dimension(0),dout.dimension(1)*dout.dimension(2)*dout.dimension(3)});
            }else
            {
                dout_2=dout;
            }
            if constexpr (dev == Device::CPU)
            {
                dbeta=dout_2.sum(Eigen::array<int, 1> {1});
                dgamma=(x_n*dout_2).sum(Eigen::array<int, 1> {1});
                Tensor2xf dx_n=one_to_two(gamma,dout_2)*dout_2;
                dx_c=dx_n/one_to_two(std,dout_2);
                Tensor1xf dstd=-((dx_n*x_c)/(one_to_two(std,dout_2)*one_to_two(std,dout_2))).sum(Eigen::array<int, 1> {1});
                Tensor1xf dvar=0.5*dstd/std;
                dx_c+=(zero_to_two(2.0/batch_size,dout_2))*x_c*one_to_two(dvar,dout_2);
                
                dm_u=dx_c.sum(Eigen::array<int, 1> {1});

                return TensorsWithNames<TensorType>{ {"dX", (dx_c-one_to_two(dm_u,dout_2)/zero_to_two(batch_size,dout_2)).reshape(x.dimensions())}};
            
            }
            else if (dev == Device::NON_OPTIMIZE)
            { 
                dbeta=sum(Eigen::array<int, 1> {1},dout_2);
                dgamma=sum(Eigen::array<int, 1> {1},mul(x_n,dout_2));
                Tensor2xf dx_n=mul(one_to_two(gamma,dout_2),dout_2);
                dx_c=div(dx_n,one_to_two(std,dout_2));
                Tensor1xf dstd=-sum(Eigen::array<int, 1> {1},div(mul(dx_n,x_c),mul(one_to_two(std,dout_2),one_to_two(std,dout_2))));

                Tensor1xf dvar=div(mul(zero_to_one(0.5,dstd),dstd),std);
                dx_c=add(dx_c,mul(mul(zero_to_two(2.0/batch_size,dout_2),x_c),one_to_two(dvar,dout_2)));
                dm_u=sum(Eigen::array<int, 1> {1},dx_c);
 
            }
            return TensorsWithNames<TensorType>{ {"dX", (sub(dx_c,div(one_to_two(dm_u,dout_2),zero_to_two(batch_size,dout_2)))).reshape(x.dimensions())}};
        }
        Tensor2xf one_to_two(Tensor1xf x_1,Tensor2xf x)
        {
            Tensor2xf x_2(x.dimensions());
            for(int i=0;i<x_2.dimension(0);i++){
                for(int j=0;j<x_2.dimension(1);j++){
                    x_2(i,j)=x_1(i);
                }
            }
            return x_2;  
        }
        Tensor2xf zero_to_two(float x_0,Tensor2xf x)
        {
            Tensor2xf x_2(x.dimensions());
            x_2.setConstant(x_0);
            return x_2;  
        }
        Tensor1xf zero_to_one(float x_0,Tensor1xf x)
        {
            Tensor1xf x_1(x.dimensions());
            x_1.setConstant(x_0);
            return x_1;  
        }
        
    };
}
