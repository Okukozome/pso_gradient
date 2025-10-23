#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include<torch/torch.h>

#include <float.h>
#include <iostream>

using Tensor = torch::Tensor;
using AutogradContext = torch::autograd::AutogradContext;
using variable_list = torch::autograd::variable_list;  //std::vector<at::Tensor>
using Variable = at::Tensor;

Tensor safeInv(Tensor& Ten, const double eps=1E-8)
{
    return Ten/(Ten*Ten+eps);
};

class gradSvd : public torch::autograd::Function<gradSvd>
{

public:

    static variable_list forward(AutogradContext *ctx, Variable var) 
    {
	   
	Tensor U, S, Vt;
        
	double scl = 1.0;

        try
	{	
            std::tie(U, S, Vt) = torch::linalg_svd(var); 
	}
	catch(const std::exception &er)
	{
	    std::cerr << "Conventional SVD attempt failed: " << er.what() << std::endl;

            try 
            {    
	        var = var + 1.0e-3 * torch::ones_like(var, var.options()) * var.abs().max();	
		std::tie(U, S, Vt) = torch::linalg_svd(var);
            }
	    catch(const std::exception &er)
            {
		std::cerr << "Regularized SVD attempt failed: " << er.what() << std::endl;    

		std::cout<<"SVD-3"<<std::endl;    

	        var = 10.0 * (var + 0.7 * (torch::rand_like(var, var.options()) - torch::rand_like(var, var.options()) + torch::ones_like(var)) * var.abs().max());	
		std::tie(U, S, Vt) = torch::linalg_svd(var); 
	    }	    
	}
        
        std::vector<Tensor> usv;
        usv.resize(3);

	double eps = 1e-9;
        auto mask = S.abs() > eps;
        S = (S + torch::ones_like(S) * eps/S.abs().max())/scl;

        //auto keep_mask = (S > S.max() * eps);
        //S = S * keep_mask;

        usv[0] = U;
        usv[1] = S;
        usv[2] = Vt.t();
        
        ctx->save_for_backward(usv);
        
        return usv;
    }

    static variable_list backward(AutogradContext *ctx, const variable_list& grad_output) 
    {
        using namespace torch::indexing;

        auto var = ctx->get_saved_variables();
        
        auto U = var[0];
        auto S = var[1];
        auto V = var[2];//.transpose(0, 1);
        
        auto Vt = at::transpose(V, 0, 1);
        auto Ut = at::transpose(U, 0, 1);

        int M = U.size(0);
        int N = V.size(0);
        int NS = S.size(0);
        

        auto St = at::transpose(S.expand({S.size(0), S.size(0)}), 0, 1); 
        auto F = S-St;
        F = safeInv(F);
        F.fill_diagonal_(0.0);
        //std::cout << F << std::endl;
        auto G = S + St;
        if (G.dtype() == torch::kF64) G.fill_diagonal_(DBL_MAX);
	else G.fill_diagonal_(FLT_MAX);
        G = 1.0/G;
        //G.fill_diagonal_(0.0);
       // std::cout << G << std::endl;

        auto dU = grad_output[0];
        auto dS = grad_output[1];
        auto dV = grad_output[2];

        auto UdU = Ut.mm(dU);
        auto VdV = Vt.mm(dV);
        

        dS = torch::diag(dS);
        
        auto Toptions = U.options();

        auto Su = (F+G).mul(UdU - UdU.transpose(0, 1))/2.0;
        auto Sv = (F-G).mul(VdV - VdV.transpose(0, 1))/2.0;

        auto dA = at::mm((Su + Sv + dS), Vt);
        dA = at::mm(U, dA);

        if (M > NS)
        {
            auto A = torch::eye(M, Toptions) - U.mm(Ut);
            auto B = dU.mm(torch::diag(1.0/S));
            B = B.mm(Vt);

            dA = dA + A.mm(B);
        }
        if (N > NS)
        {
            auto A = U.mm(torch::diag(1.0/S));
            A = A.mm(at::transpose(dV, 0, 1));
            auto B = torch::eye(N, Toptions) - V.mm(Vt);

            dA = dA + A.mm(B);
        }

        return {dA};
    }
};

std::tuple<Tensor, Tensor, Tensor> autograd_svd(const Tensor& Ten)
{
    auto vecTen = gradSvd::apply<gradSvd>(Ten);

    return std::make_tuple(vecTen[0], vecTen[1], vecTen[2].transpose(0, 1));
}
#endif
