#include <math.h>

#include "globalconfig.h"
#include "localconfig.h"
#include "stringname.h"
#include "autograd.h"
#include "CornerContraction.h"

/*
C  ---- 0    C  1 ----     C  0           C    1      E  ---- 0     E  2 ----    E   0  1  2    E  -------
   |                 |        |                |         |--- 1        1 ---|        |  |  |       |  |  |
   1                 0        ---- 1      0 ----         ---- 2        0 ----        -------       2  1  0

*/

#if !defined __STEP_CTMRG_H
#define __STEP_CTMRG_H

class StepCTMRG : public CornerContraction, public torch::autograd::Function<StepCTMRG>
{
    public:
	static variable_list forward(AutogradContext *ctx, const BLASINT &chi, const int &lenY, const int &lenX, torch::Tensor &STbt, torch::Tensor &STClt, torch::Tensor &STClb, torch::Tensor &STCrt, torch::Tensor &STCrb, torch::Tensor &STEl, torch::Tensor &STEr, torch::Tensor &STEt, torch::Tensor &STEb);    

        static variable_list backward(AutogradContext *ctx, const variable_list &gradin);

        std::chrono::milliseconds oneIteration(const BLASINT &chi, double* prot, const int &lenY, const int &lenX, std::vector<torch::Tensor> &bt, std::vector<torch::Tensor> &Clt, std::vector<torch::Tensor> &Clb, std::vector<torch::Tensor> &Crt, std::vector<torch::Tensor> &Crb, std::vector<torch::Tensor> &El, std::vector<torch::Tensor> &Er, std::vector<torch::Tensor> &Et, std::vector<torch::Tensor> &Eb);

    private:
	std::tuple<torch::Tensor, torch::Tensor> getProjectionSVDThree(const char &dir, const BLASINT &trDim, torch::Tensor* bt[4], torch::Tensor* Ct[4], torch::Tensor* E[8]);
	
        std::tuple<torch::Tensor, torch::Tensor> getProjectionSVDOne(const char &dir, const BLASINT &trDim, torch::Tensor* bt[4], torch::Tensor* Ct[4], torch::Tensor* E[8]);
        
        std::tuple<torch::Tensor, torch::Tensor> getProjectionSVDHalf(const char &dir, const BLASINT &trDim, torch::Tensor* bt[4], torch::Tensor* Ct[4], torch::Tensor* E[8]);
};

variable_list StepCTMRG::forward(AutogradContext *ctx, const BLASINT &chi, const int &lenY, const int &lenX, torch::Tensor &STbt, torch::Tensor &STClt, torch::Tensor &STClb, torch::Tensor &STCrt, torch::Tensor &STCrb, torch::Tensor &STEl, torch::Tensor &STEr, torch::Tensor &STEt, torch::Tensor &STEb)
{
    ctx->save_for_backward({STbt, STClt, STClb, STCrt, STCrb, STEl, STEr, STEt, STEb});
    ctx->saved_data["chi"] = chi;
    ctx->saved_data["lenY"] = lenY;
    ctx->saved_data["lenX"] = lenX;

    double prot[4];

    auto bt = torch::split(STbt, DB*DB, 0);

    auto clt = torch::split(STClt, chi, 0);
    auto clb = torch::split(STClb, chi, 0);
    auto crt = torch::split(STCrt, chi, 0); 
    auto crb = torch::split(STCrb, chi, 0);

    auto el = torch::split(STEl, chi, 0);
    auto er = torch::split(STEr, chi, 0);
    auto et = torch::split(STEt, chi, 0);
    auto eb = torch::split(STEb, chi, 0);

    torch::NoGradGuard no_guard;

    StepCTMRG SC;
    SC.oneIteration(chi, prot, lenY, lenX, bt, clt, clb, crt, crb, el, er, et, eb);  

    return {torch::cat(clt, 0), torch::cat(clb, 0), torch::cat(crt, 0), torch::cat(crb, 0), torch::cat(el, 0), torch::cat(er, 0), torch::cat(et, 0), torch::cat(eb, 0)};
}

variable_list StepCTMRG::backward(AutogradContext *ctx, const variable_list &gradin)
{
    int chi = ctx->saved_data["chi"].toInt();
    int lenY = ctx->saved_data["lenY"].toInt();
    int lenX = ctx->saved_data["lenX"].toInt();

    auto sv = ctx->get_saved_variables();

    double prot[4];

    auto dSTbt = sv[0].detach().set_requires_grad(sv[0].requires_grad());

    auto dSTClt = sv[1].detach().set_requires_grad(sv[1].requires_grad());
    auto dSTClb = sv[2].detach().set_requires_grad(sv[2].requires_grad());
    auto dSTCrt = sv[3].detach().set_requires_grad(sv[3].requires_grad());
    auto dSTCrb = sv[4].detach().set_requires_grad(sv[4].requires_grad());

    auto dSTEl = sv[5].detach().set_requires_grad(sv[5].requires_grad());
    auto dSTEr = sv[6].detach().set_requires_grad(sv[6].requires_grad());
    auto dSTEt = sv[7].detach().set_requires_grad(sv[7].requires_grad());
    auto dSTEb = sv[8].detach().set_requires_grad(sv[8].requires_grad());

    torch::AutoGradMode enable_grad(true);

    auto bt = torch::split(dSTbt, DB*DB, 0);

    auto clt = torch::split(dSTClt, chi, 0);
    auto clb = torch::split(dSTClb, chi, 0);
    auto crt = torch::split(dSTCrt, chi, 0);
    auto crb = torch::split(dSTCrb, chi, 0);

    auto el = torch::split(dSTEl, chi, 0);
    auto er = torch::split(dSTEr, chi, 0);
    auto et = torch::split(dSTEt, chi, 0);
    auto eb = torch::split(dSTEb, chi, 0);

    StepCTMRG SC;
    SC.oneIteration(chi, prot, lenY, lenX, bt, clt, clb, crt, crb, el, er, et, eb);

    auto STClt = torch::cat(clt, 0);
    auto STClb = torch::cat(clb, 0);
    auto STCrt = torch::cat(crt, 0);
    auto STCrb = torch::cat(crb, 0);
    auto STEl = torch::cat(el, 0);
    auto STEr = torch::cat(er, 0);
    auto STEt = torch::cat(et, 0);
    auto STEb = torch::cat(eb, 0);

    torch::autograd::backward({STClt, STClb, STCrt, STCrb, STEl, STEr, STEt, STEb}, gradin);

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), dSTbt.grad(), dSTClt.grad(), dSTClb.grad(), dSTCrt.grad(), dSTCrb.grad(), dSTEl.grad(), dSTEr.grad(), dSTEt.grad(), dSTEb.grad()};
}

std::chrono::milliseconds StepCTMRG::oneIteration(const BLASINT &chi, double* prot, const int &lenY, const int &lenX, std::vector<torch::Tensor> &bt, std::vector<torch::Tensor> &Clt, std::vector<torch::Tensor> &Clb, std::vector<torch::Tensor> &Crt, std::vector<torch::Tensor> &Crb, std::vector<torch::Tensor> &El, std::vector<torch::Tensor> &Er, std::vector<torch::Tensor> &Et, std::vector<torch::Tensor> &Eb)
{
    assert(chi > 0);

    std::cout.precision(10);

    auto timer_start = std::chrono::high_resolution_clock::now();

/*to left*/ 

    double protl = 1.0;

    for (int x = 0; x < lenX; x++)
    {
        const int xp = (x+1)%lenX;

        std::tuple<torch::Tensor, torch::Tensor> pqt[lenY]; 

	for (int y = 0; y < lenY; y++)
	{
	    const int yp = (y+1)%lenY;
	        
	    const int yx = y*lenX+x;
	    const int yxp = y*lenX+xp;	
	    const int ypx = yp*lenX+x;
            const int ypxp = yp*lenX+xp;

            const int r0 = yx;
	    const int r1 = yxp;
	    const int r2 = ypx;
	    const int r3 = ypxp;	

            torch::Tensor* pbt[4]; 
	    pbt[0] = &bt[r0];
	    pbt[1] = &bt[r1];
	    pbt[2] = &bt[r2];
	    pbt[3] = &bt[r3];

	    torch::Tensor* pct[4];
	    pct[0] = &Clb[r0];	
            pct[1] = &Crb[r1];
	    pct[2] = &Clt[r2];
	    pct[3] = &Crt[r3];

	    torch::Tensor* pet[8];
	    pet[0] = &Eb[r0];
	    pet[1] = &Eb[r1];
	    pet[2] = &El[r0];
	    pet[3] = &Er[r1];
	    pet[4] = &El[r2];
	    pet[5] = &Er[r3];
	    pet[6] = &Et[r2];
	    pet[7] = &Et[r3];

	    pqt[y] = getProjectionSVDOne('L', chi, pbt, pct, pet); 
	}

	double pro = 1.0;

        for (int y = 0; y < lenY; y++)
	{
	    const int ym = (y-1+lenY)%lenY;

	    const int yx = y*lenX+x;
            const int yxp = y*lenX+xp;

	    torch::Tensor EC = torch::tensordot(Et[yx], Clt[yx], {2}, {0}).permute({0, 2, 1}).contiguous();
	    auto tEC = EC.view({EC.size(0), EC.size(1)*EC.size(2)});
            auto tECpq =  torch::tensordot(tEC, std::get<1>(pqt[y]), {1}, {1});
            auto s = torch::abs(tECpq).max();
	    Clt[yxp] = torch::div(tECpq, s); 
            pro *= s.item().toFloat();

            auto CE = torch::tensordot(Clb[yx], Eb[yx], {1}, {0});
	    auto tCE = CE.view({CE.size(0)*CE.size(1), CE.size(2)}); 
            auto pqtCE = torch::tensordot(std::get<0>(pqt[ym]), tCE, {0}, {0});
	    auto r = torch::abs(pqtCE).max();
	    Clb[yxp] = torch::div(pqtCE, r);
	    pro *= r.item().toFloat();

	    auto chiq0 = El[yx].size(2);
	    auto chiq1 = bt[yx].size(1);
            torch::Tensor &A = std::get<1>(pqt[ym]);
	    auto tA = A.view({A.size(0), chiq0, chiq1});
	    auto EA = torch::tensordot(El[yx], tA, {2}, {1}).permute({0, 2, 1, 3}).contiguous();
	    auto vEA = EA.view({EA.size(0), EA.size(1), EA.size(2)*EA.size(3)});
	    auto vbt = bt[yx].view({bt[yx].size(0)*bt[yx].size(1), bt[yx].size(2), bt[yx].size(3)});
	    auto vbtA = torch::tensordot(vbt, vEA, {0}, {2}).permute({2, 1, 0, 3}).contiguous();
	    auto yy = vbtA.view({vbtA.size(0)*vbtA.size(1), vbtA.size(2), vbtA.size(3)});
            auto ddm = torch::tensordot(std::get<0>(pqt[y]), yy, {0}, {0});
	    auto el = torch::abs(ddm).max();	
            El[yxp] = torch::div(ddm, el);
            pro *= el.item().toFloat();
    	}
          
	protl *= pro;

//        std::cout<<  " pro = " << pro << std::endl;	   
    }

/*toRight*/

    double protr = 1.0;

    for (int x = lenX-1; x >= 0; x--)
    {
	const int xm = (x-1+lenX)%lenX;
           
        std::tuple<torch::Tensor, torch::Tensor> pqt[lenY];

        for (int y = 0; y < lenY; y++)
	{
            const int yp = (y+1)%lenY;
               
	    const int yx = y*lenX+x;
	    const int yxm = y*lenX+xm;
	    const int ypx = yp*lenX+x;
	    const int ypxm = yp*lenX+xm;

	    const int r0 = yxm;
	    const int r1 = yx;
	    const int r2 = ypxm;
	    const int r3 = ypx;

	    torch::Tensor* pbt[4];
            pbt[0] = &bt[r0];
            pbt[1] = &bt[r1];
            pbt[2] = &bt[r2];
            pbt[3] = &bt[r3];

	    torch::Tensor* pct[4];
	    pct[0] = &Clb[r0];	
	    pct[1] = &Crb[r1];
	    pct[2] = &Clt[r2];
	    pct[3] = &Crt[r3];	
                
            torch::Tensor* pet[8];
            pet[0] = &Eb[r0];	
	    pet[1] = &Eb[r1];
	    pet[2] = &El[r0];
	    pet[3] = &Er[r1];
	    pet[4] = &El[r2];
	    pet[5] = &Er[r3];
	    pet[6] = &Et[r2];
	    pet[7] = &Et[r3];

	    pqt[y] = getProjectionSVDOne('R', chi, pbt, pct, pet);	
	}	    

	double pro = 1.0;

        for (int y = 0; y < lenY; y++) 
	{
            const int ym = (y-1+lenY)%lenY;
                
            const int yx = y*lenX+x;
            const int yxm = y*lenX+xm;

            auto CE = torch::tensordot(Crt[yx], Et[yx], {1}, {0}).permute({1, 0, 2}).contiguous();
	    auto vCE = CE.view({CE.size(0)*CE.size(1), CE.size(2)});
            auto pqCE = torch::tensordot(std::get<1>(pqt[y]), vCE, {1}, {0});
	    auto s = torch::abs(pqCE).max();
	    Crt[yxm] = torch::div(pqCE, s);
	    pro *= s.item().toFloat();
                
            auto EC = torch::tensordot(Eb[yx], Crb[yx], {2}, {0});
	    auto vEC = EC.view({EC.size(0), EC.size(1)*EC.size(2)});
            auto ECpq = torch::tensordot(vEC, std::get<0>(pqt[ym]), {1}, {0});
            auto r = torch::abs(ECpq).max();
	    Crb[yxm] = torch::div(ECpq, r);
	    pro *= r.item().toFloat();	

	    const BLASINT chiq0 = bt[yx].size(1);
	    const BLASINT chiq1 = Er[yx].size(0);

	    torch::Tensor &A = std::get<1>(pqt[ym]);
            auto vA = A.view({A.size(0), chiq0, chiq1});
	    auto vbt = bt[yx].view({bt[yx].size(0), bt[yx].size(1)*bt[yx].size(2), bt[yx].size(3)}).permute({0, 2, 1});
            auto AE = torch::tensordot(vA, Er[yx], {2}, {0});
            auto vAE = AE.view({AE.size(0), AE.size(1)*AE.size(2), AE.size(3)}).permute({0, 2, 1});
	    auto AEbt = torch::tensordot(vAE, vbt, {2}, {2}).permute({0, 2, 3, 1}).contiguous();	
            auto vAEbt = AEbt.view({AEbt.size(0), AEbt.size(1), AEbt.size(2)*AEbt.size(3)});
	    auto ddf = torch::tensordot(vAEbt, std::get<0>(pqt[y]), {2}, {0});
            auto el = torch::abs(ddf).max();
	    Er[yxm] = torch::div(ddf, el);
	    pro *= el.item().toFloat();
	}

	protr *= pro;

//        std::cout << " pro = " << pro << std::endl;
    }	
     
/*to bottom*/

    double protb = 1.0;

    for (int y = 0; y < lenY; y++)
    {
        const int yp = (y+1)%lenY;
         
	std::tuple<torch::Tensor, torch::Tensor> pqt[lenX];

        for (int x = 0; x < lenX; x++)	    
	{
	    const int xp = (x+1)%lenX;

            const int yx = y*lenX+x;
            const int yxp = y*lenX+xp;
            const int ypx = yp*lenX+x;
            const int ypxp = yp*lenX+xp;

	    const int r0 = yx;
	    const int r1 = yxp;
	    const int r2 = ypx;
	    const int r3 = ypxp;

	    torch::Tensor* pbt[4];
            pbt[0] = &bt[r0];
            pbt[1] = &bt[r1];
            pbt[2] = &bt[r2];	
	    pbt[3] = &bt[r3];	

	    torch::Tensor* pct[4];
	    pct[0] = &Clb[r0];
	    pct[1] = &Crb[r1];
	    pct[2] = &Clt[r2];
	    pct[3] = &Crt[r3];

	    torch::Tensor* pet[8];
	    pet[0] = &Eb[r0];
	    pet[1] = &Eb[r1];
	    pet[2] = &El[r0];
	    pet[3] = &Er[r1];
	    pet[4] = &El[r2];
	    pet[5] = &Er[r3];
	    pet[6] = &Et[r2];
	    pet[7] = &Et[r3];

	    pqt[x] = getProjectionSVDOne('D', chi, pbt, pct, pet); 
	}

        double pro = 1.0;

	for (int x = 0; x < lenX; x++)
	{
	    const int xm = (x-1+lenX)%lenX;
            const int yx = y*lenX+x;
            const int ypx = yp*lenX+x;
               
            torch::Tensor EC = torch::tensordot(El[yx], Clb[yx], {2}, {0}).permute({0, 2, 1}).contiguous();
	    auto vEC = EC.view({EC.size(0), EC.size(1)*EC.size(2)});
	    auto ECpq = torch::tensordot(vEC, std::get<0>(pqt[xm]), {1}, {0});
	    auto s = torch::abs(ECpq).max();
	    Clb[ypx] = torch::div(ECpq, s);
            pro *= s.item().toFloat();
	        
	    torch::Tensor CE = torch::tensordot(Crb[yx], Er[yx], {1}, {0});
            auto vCE = CE.view({CE.size(0)*CE.size(1), CE.size(2)}); 
	    auto pqCE = torch::tensordot(std::get<1>(pqt[x]), vCE, {1}, {0});
	    auto r = torch::abs(pqCE).max();
	    Crb[ypx] = torch::div(pqCE, r);
	    pro *= r.item().toFloat();

	    const BLASINT chiq0 = Eb[yx].size(0);
	    const BLASINT chiq1 = bt[yx].size(0);

            torch::Tensor &A = std::get<1>(pqt[xm]);
            auto vA = A.view({A.size(0), chiq0, chiq1});
            torch::Tensor AE = torch::tensordot(vA, Eb[yx], {1}, {0});
	    auto vAE = AE.view({AE.size(0), AE.size(1)*AE.size(2), AE.size(3)});
	    auto vbt = bt[yx].view({bt[yx].size(0)*bt[yx].size(1), bt[yx].size(2), bt[yx].size(3)});
	    auto AEbt = torch::tensordot(vAE, vbt, {1}, {0});
            auto vAEbt = AEbt.view({AEbt.size(0), AEbt.size(1)*AEbt.size(2), AEbt.size(3)});
	    auto ggm = torch::tensordot(vAEbt, std::get<0>(pqt[x]), {1}, {0});
	    auto el = torch::abs(ggm).max();
	    Eb[ypx] = torch::div(ggm, el);
            pro *= el.item().toFloat();
	}

	protb *= pro;

//	std::cout<<  " pro = " << pro << std::endl;
    }
       
/*to top*/	

    double prott = 1.0;

    for (int y = lenY-1; y >= 0; y--)
    {
	const int ym = (y-1+lenY)%lenY;

        std::tuple<torch::Tensor, torch::Tensor> pqt[lenX];

        for (int x = 0; x < lenX; x++)
	{
	    const int xp = (x+1)%lenX;
            const int yx = y*lenX+x;
            const int ymx = ym*lenX+x;
            const int ymxp = ym*lenX+xp;
            const int yxp = y*lenX+xp;		

	    const int r0 = ymx;
	    const int r1 = ymxp;
	    const int r2 = yx;
	    const int r3 = yxp;

	    torch::Tensor* pbt[4];
	    pbt[0] = &bt[r0];
	    pbt[1] = &bt[r1];
	    pbt[2] = &bt[r2];
	    pbt[3] = &bt[r3];
	        
	    torch::Tensor* pct[4];
	    pct[0] = &Clb[r0];
	    pct[1] = &Crb[r1];
	    pct[2] = &Clt[r2];
	    pct[3] = &Crt[r3];

	    torch::Tensor* pet[8];
	    pet[0] = &Eb[r0];
	    pet[1] = &Eb[r1];
	    pet[2] = &El[r0];
	    pet[3] = &Er[r1];
	    pet[4] = &El[r2];	
	    pet[5] = &Er[r3];
	    pet[6] = &Et[r2];
	    pet[7] = &Et[r3];

	    pqt[x] = getProjectionSVDOne('U', chi, pbt, pct, pet);
	}

	double pro = 1.0;
            
        for (int x = 0; x < lenX; x++)
	{
	    const int xm = (x-1+lenX)%lenX;

	    const int yx = y*lenX+x;
	    const int ymx = ym*lenX+x;

	    auto CE = torch::tensordot(Clt[yx], El[yx], {1}, {0}).permute({1, 0, 2}).contiguous();
	    auto vCE = CE.view({CE.size(0)*CE.size(1), CE.size(2)});
            auto pqCE = torch::tensordot(std::get<0>(pqt[xm]), vCE, {0}, {0});
	    auto s = torch::abs(pqCE).max();
	    Clt[ymx] = torch::div(pqCE, s);
            pro *= s.item().toFloat(); 

            auto EC = torch::tensordot(Er[yx], Crt[yx], {2}, {0});
	    auto vEC = EC.view({EC.size(0), EC.size(1)*EC.size(2)});
            auto ECpq = torch::tensordot(vEC, std::get<1>(pqt[x]), {1}, {1});
	    auto r = torch::abs(ECpq).max();
	    Crt[ymx] = torch::div(ECpq, r);
	    pro *= r.item().toFloat();

            const BLASINT chip0 = bt[yx].size(2);
	    const BLASINT chip1 = Et[yx].size(0);

            torch::Tensor &A = std::get<0>(pqt[x]);
            auto vA = A.view({chip0, chip1, A.size(1)});
            auto AE = torch::tensordot(vA, Et[yx], {1}, {0}).permute({0, 2, 3, 1}).contiguous();
	    auto vAE = AE.view({AE.size(0)*AE.size(1), AE.size(2), AE.size(3)});
            auto vbt = bt[yx].view({bt[yx].size(0), bt[yx].size(1), bt[yx].size(2)*bt[yx].size(3)});
            auto AEbt = torch::tensordot(vAE, vbt, {0}, {2}).permute({2, 0, 1, 3}).contiguous();
	    auto vAEbt = AEbt.view({AEbt.size(0)*AEbt.size(1), AEbt.size(2), AEbt.size(3)});
            auto ggm = torch::tensordot(vAEbt, std::get<1>(pqt[xm]), {0}, {1});
	    auto el = torch::abs(ggm).max();
	    Et[ymx] = torch::div(ggm, el);
	    pro *= el.item().toFloat();
        }

	prott *= pro;

//        std::cout<<  " pro = " << pro << std::endl;	    
    }
      
//    std::cout << "protl = " << protl << ", protr =  " << protr << ";  protb = " << protb << ", prott = " <<	prott << std::endl;

    prot[0] = protl;
    prot[1] = protr;
    prot[2] = protb;
    prot[3] = prott;

    auto timer_end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    return duration;
}

/*calculating projection operators  P, Q :  from bottom to top or from left to right */

std::tuple<torch::Tensor, torch::Tensor> StepCTMRG::getProjectionSVDThree(const char &dir, const BLASINT &trDim, torch::Tensor* bt[4], torch::Tensor* Ct[4], torch::Tensor* E[8])
{
#ifdef DEBUG_GPJT_TIME
    std::cout<< std::endl << "**************Start getProjection! ******************" << std::endl;
    auto timer_start = std::chrono::high_resolution_clock::now();
#endif

    assert(dir == 'L' || dir == 'R' || dir == 'U' || dir == 'D');

    torch::Tensor lbc = LBContraction(*Ct[0], *E[2], *E[0], *bt[0]);
    torch::Tensor rbc = RBContraction(*Ct[1], *E[3], *E[1], *bt[1]);
    torch::Tensor ltc = LTContraction(*Ct[2], *E[4], *E[6], *bt[2]);
    torch::Tensor rtc = RTContraction(*Ct[3], *E[5], *E[7], *bt[3]);

    torch::Tensor AT, BT;

    if (dir == 'L')
    {
        AT = torch::tensordot(rbc, lbc, {0}, {1});
        BT = torch::tensordot(ltc, rtc, {0}, {1});
    }
    else if (dir == 'R')
    {
	AT = torch::tensordot(lbc, rbc, {1}, {0});
        BT = torch::tensordot(rtc, ltc, {1}, {0});	
    }
    else if (dir == 'U')
    {
	AT = torch::tensordot(lbc, ltc, {0}, {1});
        BT = torch::tensordot(rtc, rbc, {0}, {1});	
    }
    else
    {
	AT = torch::tensordot(ltc, lbc, {1}, {0}); 
        BT = torch::tensordot(rbc, rtc, {1}, {0});	
    }

/* AB = USV^+,   idensity matrix   I = B (AB)^(-1) A = B VS^(-1)U^+ A = BVS^{-1/2} S^{-1/2}UA = P Q 
 * Here P and Q are the projection operators, and the bond connecting them is truncated to trDim. 
 * */

    auto [Ua, Sa, Va] = autograd_svd(AT); //at::linalg_svd(AT);
    auto [Ub, Sb, Vb] = autograd_svd(BT); //at::linalg_svd(BT); 
   
    auto La = torch::einsum("a, ab->ab", {Sa, Va});
    auto Rb = torch::einsum("b ,ab->ab", {Sb, Ub});

    auto LR = torch::tensordot(La, Rb, {1}, {0});
    
    auto [Ulr, Slr, Vlr] = autograd_svd(LR); //at::linalg_svd(LR);

    auto trS = torch::pow(torch::sqrt(torch::narrow(Slr, 0, 0, trDim)), -1.0);

    auto cVlr = torch::conj_physical(torch::narrow(Vlr, 0, 0, trDim).permute({1, 0}));
    auto SVlr = torch::einsum("b, ab->ab", {trS, cVlr}); 

    torch::Tensor ptt = torch::tensordot(Rb, SVlr, {1}, {0}); 

    auto cUlr = torch::conj_physical(torch::narrow(Ulr, 1, 0, trDim).permute({1, 0}));
    auto SUlr = torch::einsum("a, ab->ab", {trS, cUlr});

    torch::Tensor qtt = torch::tensordot(SUlr, La, {1}, {0});
   
#ifdef DEBUG_GPJT_TIME
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    std::cout << "ProjectionSVDThree time is " << duration.count() << " ms\n";
#endif

    return std::tuple<torch::Tensor, torch::Tensor>(ptt, qtt); 
}

/*calculating projection operators  P, Q :  from bottom to top or from left to right */

std::tuple<torch::Tensor, torch::Tensor> StepCTMRG::getProjectionSVDOne(const char &dir, const BLASINT &trDim, torch::Tensor* bt[4], torch::Tensor* Ct[4], torch::Tensor* E[8])
{
#ifdef DEBUG_GPJT_TIME
    std::cout<< std::endl << "**************Start getProjection! ******************" << std::endl;
    auto timer_start = std::chrono::high_resolution_clock::now();
#endif

    assert(dir == 'L' || dir == 'R' || dir == 'U' || dir == 'D');

    torch::Tensor lbc = LBContraction(*Ct[0], *E[2], *E[0], *bt[0]);
    torch::Tensor rbc = RBContraction(*Ct[1], *E[3], *E[1], *bt[1]);
    torch::Tensor ltc = LTContraction(*Ct[2], *E[4], *E[6], *bt[2]);
    torch::Tensor rtc = RTContraction(*Ct[3], *E[5], *E[7], *bt[3]);

    torch::Tensor AT, BT;

    if (dir == 'L')
    {
        AT = torch::tensordot(rbc, lbc, {0}, {1});
        BT = torch::tensordot(ltc, rtc, {0}, {1});
    }
    else if (dir == 'R')
    {
	AT = torch::tensordot(lbc, rbc, {1}, {0});
        BT = torch::tensordot(rtc, ltc, {1}, {0});	
    }
    else if (dir == 'U')
    {
	AT = torch::tensordot(lbc, ltc, {0}, {1});
        BT = torch::tensordot(rtc, rbc, {0}, {1});	
    }
    else
    {
	AT = torch::tensordot(ltc, lbc, {1}, {0}); 
        BT = torch::tensordot(rbc, rtc, {1}, {0});	
    }

/* AB = USV^+,   idensity matrix   I = B (AB)^(-1) A = B VS^(-1)U^+ A = BVS^{-1/2} S^{-1/2}UA = P Q 
 * Here P and Q are the projection operators, and the bond connecting them is truncated to trDim. 
 * */

    auto LR = torch::tensordot(AT, BT, {1}, {0});
   
#ifdef DEBUG_GPJT_TIME    
    auto timer_ssvd = std::chrono::high_resolution_clock::now();
#endif    

    auto [Ulr, Slr, Vlr] = autograd_svd(LR);//at::linalg_svd(LR);
					    
#ifdef DEBUG_GPJT_TIME    
    auto timer_esvd = std::chrono::high_resolution_clock::now();
    auto svdtime = std::chrono::duration_cast<std::chrono::milliseconds>(timer_esvd - timer_ssvd);
    std::cout << "SVD time is " << svdtime.count() << "ms\n"<<std::endl;
#endif

    auto trS = torch::pow(torch::sqrt(torch::narrow(Slr, 0, 0, trDim)), -1.0);

    auto cVlr = torch::conj_physical(torch::narrow(Vlr, 0, 0, trDim).permute({1, 0}));
    auto SVlr = torch::einsum("b, ab->ab", {trS, cVlr}); 

    torch::Tensor ptt = torch::tensordot(BT, SVlr, {1}, {0}); 

    auto cUlr = torch::conj_physical(torch::narrow(Ulr, 1, 0, trDim).permute({1, 0}));
    auto SUlr = torch::einsum("a, ab->ab", {trS, cUlr});

    torch::Tensor qtt = torch::tensordot(SUlr, AT, {1}, {0});
   
#ifdef DEBUG_GPJT_TIME
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    std::cout << "ProjectionSVDOne time is " << duration.count() << " ms\n";
#endif

    return std::tuple<torch::Tensor, torch::Tensor>(ptt, qtt); 
}

/*calculating projection operators  P, Q :  from bottom to top or from left to right */

std::tuple<torch::Tensor, torch::Tensor> StepCTMRG::getProjectionSVDHalf(const char &dir, const BLASINT &trDim, torch::Tensor* bt[4], torch::Tensor* Ct[4], torch::Tensor* E[8])
{
#ifdef DEBUG_GPJT_TIME
    std::cout<< std::endl << "**************Start getProjection! ******************" << std::endl;
    auto timer_start = std::chrono::high_resolution_clock::now();
#endif

    assert(dir == 'L' || dir == 'R' || dir == 'U' || dir == 'D');

    torch::Tensor AT, BT;

    if (dir == 'L')
    {
	AT = LBContraction(*Ct[0], *E[2], *E[0], *bt[0]).permute({1, 0}).contiguous();
	BT = LTContraction(*Ct[2], *E[4], *E[6], *bt[2]).permute({1, 0}).contiguous();
    }
    else if (dir == 'R')
    {
        AT = RBContraction(*Ct[1], *E[3], *E[1], *bt[1]); //torch::tensordot(II, rbc, {1}, {0});
        BT = RTContraction(*Ct[3], *E[5], *E[7], *bt[3]); //torch::tensordot(rtc, II, {1}, {0});
    }
    else if (dir == 'U')
    {
        AT = LTContraction(*Ct[2], *E[4], *E[6], *bt[2]).permute({1, 0}).contiguous();  //torch::tensordot(II, ltc, {0}, {1});
        BT = RTContraction(*Ct[3], *E[5], *E[7], *bt[3]).permute({1, 0}).contiguous();  //torch::tensordot(rtc, II, {0}, {1});
    }
    else
    {
        AT = LBContraction(*Ct[0], *E[2], *E[0], *bt[0]); //torch::tensordot(II, lbc, {1}, {0});
        BT = RBContraction(*Ct[1], *E[3], *E[1], *bt[1]); //torch::tensordot(rbc, II, {1}, {0});
    }

/* AB = USV^+,   idensity matrix   I = B (AB)^(-1) A = B VS^(-1)U^+ A = BVS^{-1/2} S^{-1/2}UA = P Q 
 * Here P and Q are the projection operators, and the bond connecting them is truncated to trDim. 
 * */

    auto LR = torch::tensordot(AT, BT, {1}, {0});
   
#ifdef DEBUG_GPJT_TIME    
    auto timer_ssvd = std::chrono::high_resolution_clock::now();
#endif    

    auto [Ulr, Slr, Vlr] = autograd_svd(LR);//at::linalg_svd(LR);
					    
#ifdef DEBUG_GPJT_TIME    
    auto timer_esvd = std::chrono::high_resolution_clock::now();
    auto svdtime = std::chrono::duration_cast<std::chrono::milliseconds>(timer_esvd - timer_ssvd);
    std::cout << "SVD time is " << svdtime.count() << "ms\n"<<std::endl;
#endif

    auto trS = torch::pow(torch::sqrt(torch::narrow(Slr, 0, 0, trDim)), -1.0);

    auto cVlr = torch::conj_physical(torch::narrow(Vlr, 0, 0, trDim).permute({1, 0}));
    auto SVlr = torch::einsum("b, ab->ab", {trS, cVlr}); 

    torch::Tensor ptt = torch::tensordot(BT, SVlr, {1}, {0}); 

    auto cUlr = torch::conj_physical(torch::narrow(Ulr, 1, 0, trDim).permute({1, 0}));
    auto SUlr = torch::einsum("a, ab->ab", {trS, cUlr});

    torch::Tensor qtt = torch::tensordot(SUlr, AT, {1}, {0});
   
#ifdef DEBUG_GPJT_TIME
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    std::cout << "ProjectionSVDOne time is " << duration.count() << " ms\n";
#endif

    return std::tuple<torch::Tensor, torch::Tensor>(ptt, qtt); 
}

variable_list TT(const BLASINT &chi, const int &lenY, const int &lenX, torch::Tensor &STbt, torch::Tensor &STClt, torch::Tensor &STClb, torch::Tensor &STCrt, torch::Tensor &STCrb, torch::Tensor &STEl, torch::Tensor &STEr, torch::Tensor &STEt, torch::Tensor &STEb)
{
    return StepCTMRG::apply<StepCTMRG>(chi, lenY, lenX, STbt, STClt, STClb, STCrt, STCrb, STEl, STEr, STEt, STEb);
}

#endif
