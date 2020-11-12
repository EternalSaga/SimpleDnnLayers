#pragma once

#include "OpInterfaces.hpp"
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
        TensorType forwardImpl(const TensorsWithNames<TensorType> &argX,std::ofstream& outt){
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
                Tensor1xf m_u(x_2.dimension(0));
                mean_(m_u.data(),1,x_2.data(),{x_2.dimension(0),x_2.dimension(1)});

                x_c=Tensor2xf {x_2.dimensions()};
                sub_two(x_c.data(),x_2.data(),one_to_two(m_u,x_2).data(),{x_2.dimension(0),x_2.dimension(1)});

                Tensor2xf m_2(x_c.dimensions());
                mul_two(m_2.data(),x_c.data(),x_c.data(),{x_c.dimension(0),x_c.dimension(1)});
                Tensor1xf var(m_2.dimension(0));
                mean_(var.data(),1,m_2.data(),{m_2.dimension(0),m_2.dimension(1)});

                std=Tensor1xf {var.dimensions()};
                add_one(std.data(),var.data(),zero_to_one(variance_epsilon,var).data(),{var.dimension(0)});
                sqrt_one(std.data(),{std.dimension(0)});

                
                x_n=Tensor2xf {x_c.dimensions()};
                div_two(x_n.data(),x_c.data(),one_to_two(std,x_2).data(),{x_c.dimension(0),x_c.dimension(1)});

                batch_size=x_2.dimension(0);

                Tensor1xf m_ml(zero_to_one(decay,batch_Mean).dimensions());
                mul_one(m_ml.data(),zero_to_one(decay,batch_Mean).data(),batch_Mean.data(),{zero_to_one(decay,batch_Mean).dimension(0)});
                Tensor1xf m_mr(zero_to_one(1-decay,batch_Mean).dimensions());
                mul_one(m_mr.data(),zero_to_one(1-decay,batch_Mean).data(),m_u.data(),{zero_to_one(1-decay,batch_Mean).dimension(0)});
                batch_Mean=Tensor1xf {m_ml.dimensions()};
                add_one(batch_Mean.data(),m_ml.data(),m_mr.data(),{m_ml.dimension(0)});

                Tensor1xf m_bl(zero_to_one(decay,batch_Var).dimensions());
                mul_one(m_bl.data(),zero_to_one(decay,batch_Var).data(),batch_Var.data(),{zero_to_one(decay,batch_Var).dimension(0)});
                Tensor1xf m_br(zero_to_one(1-decay,batch_Var).dimensions());
                mul_one(m_br.data(),zero_to_one(1-decay,batch_Var).data(),var.data(),{zero_to_one(1-decay,batch_Var).dimension(0)});
                batch_Var=Tensor1xf {m_bl.dimensions()};
                add_one(batch_Var.data(),m_bl.data(),m_br.data(),{m_bl.dimension(0)});
            
                x_c=Tensor2xf {x_2.dimensions()};
                sub_two(x_c.data(),x_2.data(),one_to_two(batch_Mean,x_2).data(),{x_2.dimension(0),x_2.dimension(1)});

                Tensor2xf a_s(one_to_two(batch_Var,x_2).dimensions());
                add_two(a_s.data(),one_to_two(batch_Var,x_2).data(),zero_to_two(variance_epsilon,x_2).data(),{one_to_two(batch_Var,x_2).dimension(0),one_to_two(batch_Var,x_2).dimension(1)});
                sqrt_two(a_s.data(),{a_s.dimension(0),a_s.dimension(1)});
                x_n=Tensor2xf {x_c.dimensions()};
                div_two(x_n.data(),x_c.data(),a_s.data(),{x_c.dimension(0),x_c.dimension(1)});/**/ 
                
            }
            Tensor2xf m_al(one_to_two(gamma,x_2).dimensions());
            mul_two(m_al.data(),one_to_two(gamma,x_2).data(),x_n.data(),{one_to_two(gamma,x_2).dimension(0),one_to_two(gamma,x_2).dimension(1)});
            Tensor2xf a_r(m_al.dimensions());
            add_two(a_r.data(),m_al.data(),one_to_two(beta,x_2).data(),{m_al.dimension(0),m_al.dimension(1)});
            return a_r.reshape(x.dimensions());
        }

        TensorsWithNames<TensorType> backwardImpl(TensorType& dout, std::ofstream& outt)
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
                dbeta=Tensor1xf {dout_2.dimension(0)};
                sum_(dbeta.data(),1,dout_2.data(),{dout_2.dimension(0),dout_2.dimension(1)});
                
                Tensor2xf dm_2(x_n.dimensions());
                mul_two(dm_2.data(),x_n.data(),dout_2.data(),{x_n.dimension(0),x_n.dimension(1)});
                dgamma=Tensor1xf {dm_2.dimension(0)};
                sum_(dgamma.data(),1,dm_2.data(),{dm_2.dimension(0),dm_2.dimension(1)});

                Tensor2xf dx_n(one_to_two(gamma,dout_2).dimensions());
                mul_two(dx_n.data(),one_to_two(gamma,dout_2).data(),dout_2.data(),{one_to_two(gamma,dout_2).dimension(0),one_to_two(gamma,dout_2).dimension(1)});
                dx_c=Tensor2xf {dx_n.dimensions()};
                div_two(dx_c.data(),dx_n.data(),one_to_two(std,dout_2).data(),{dx_n.dimension(0),dx_n.dimension(1)});

                Tensor2xf dd_r(one_to_two(std,dout_2).dimensions());
                mul_two(dd_r.data(),one_to_two(std,dout_2).data(),one_to_two(std,dout_2).data(),{one_to_two(std,dout_2).dimension(0),one_to_two(std,dout_2).dimension(1)});
                Tensor2xf dd_l(dx_n.dimensions());
                mul_two(dd_l.data(),dx_n.data(),x_c.data(),{dx_n.dimension(0),dx_n.dimension(1)});
                Tensor2xf dd_2(dd_l.dimensions());
                div_two(dd_2.data(),dd_l.data(),dd_r.data(),{dd_l.dimension(0),dd_l.dimension(1)});
                Tensor1xf dstd(dd_2.dimension(0));
                sum_de(dstd.data(),1,dd_2.data(),{dd_2.dimension(0),dd_2.dimension(1)});
 
                Tensor1xf dm_l(zero_to_one(0.5,dstd).dimensions());
                mul_one(dm_l.data(),zero_to_one(0.5,dstd).data(),dstd.data(),{zero_to_one(0.5,dstd).dimension(0)});
                Tensor1xf dvar(dm_l.dimensions());
                div_one(dvar.data(),dm_l.data(),std.data(),{dm_l.dimension(0)});

                Tensor2xf dm_ml(zero_to_two(2.0/batch_size,dout_2).dimensions());
                mul_two(dm_ml.data(),zero_to_two(2.0/batch_size,dout_2).data(),x_c.data(),{zero_to_two(2.0/batch_size,dout_2).dimension(0),zero_to_two(2.0/batch_size,dout_2).dimension(1)});
                Tensor2xf dm_m(dm_ml.dimensions());
                mul_two(dm_m.data(),dm_ml.data(),one_to_two(dvar,dout_2).data(),{dm_ml.dimension(0),dm_ml.dimension(1)});
                add_two(dx_c.data(),dx_c.data(),dm_m.data(),{dx_c.dimension(0),dx_c.dimension(1)});

                dm_u=Tensor1xf {dx_c.dimension(0)};
                sum_(dm_u.data(),1,dx_c.data(),{dx_c.dimension(0),dx_c.dimension(1)}); /**/

            }
            Tensor2xf dd_s(one_to_two(dm_u,dout_2).dimensions());
            div_two(dd_s.data(),one_to_two(dm_u,dout_2).data(),zero_to_two(batch_size,dout_2).data(),{one_to_two(dm_u,dout_2).dimension(0),one_to_two(dm_u,dout_2).dimension(1)});
            Tensor2xf ds_r(dx_c.dimensions());
            sub_two(ds_r.data(),dx_c.data(),dd_s.data(),{dx_c.dimension(0),dx_c.dimension(1)});
            return TensorsWithNames<TensorType>{ {"dX", ds_r.reshape(x.dimensions())}};
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
        
