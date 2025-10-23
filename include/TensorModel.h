#if !defined __TENSOR_MODEL_H
#define __TENSOR_MODEL_H

#define MM_PI (3.14159265358979)

#include <fstream>
#include <stdlib.h>
#include <assert.h>

#include "TensorExpectation.h"
#include "ctmrg.h"

class TensorModel : public CTMRG, public CornerContraction, public TensorExpectation
{
    public:
	torch::Tensor LTC[MAXCST];
        torch::Tensor RTC[MAXCST];	
	torch::Tensor LBC[MAXCST];
        torch::Tensor RBC[MAXCST];

    public:
        TensorModel(const int &fidx, const BLASINT &ly, const BLASINT &lx, torch::Tensor* bt, const char &state, const BLASINT &chi);

    public:
	torch::Tensor getExpt();	
	
	void siteExpt(); 
       
	torch::Tensor bondExpt();
 
	torch::Tensor bondExptX(const BLASINT &y, const BLASINT &x, torch::Tensor &hST);

	torch::Tensor bondExptY(const BLASINT &y, const BLASINT &x, torch::Tensor &hST);

/*cross from top(left) to bottom(right)*/

	torch::Tensor bondExptCTB(const BLASINT &y, const BLASINT &x, torch::Tensor &hST);

/*cross from bottom(left) to top(right)*/

	torch::Tensor bondExptCBT(const BLASINT &y, const BLASINT &x, torch::Tensor &hST);

/*ltc, rtc, lbc, rbc*/

	void calculateECE();

    private:
        TensorModel();

        void setBondOperator(torch::Tensor &hamST, const torch::Tensor &topt);
};

TensorModel::TensorModel(const int &fidx, const BLASINT &ly, const BLASINT &lx, torch::Tensor* wt, const char &state, const BLASINT &chi) : CTMRG(fidx, ly, lx, wt, state, chi)
{
}

torch::Tensor TensorModel::getExpt()
{
    time_t time_begin = time(0);

//    siteExpt(Clb, Crb, Clt, Crt, El, Eb, Er, Et, bt, lenY, lenX); 

    time_t time_site = time(0);

    std::cout << "========== Time for siteExpt is " << time_site - time_begin << std::endl << std::endl;

    auto eg = bondExpt();

    time_t time_bond = time(0);

    std::cout << "========== Time for bondExpt is " << time_bond - time_site << std::endl << std::endl;

    return eg;
}

void TensorModel::siteExpt()
{
    const BLASINT ly = lenY;
    const BLASINT lx = lenX;

    torch::Tensor xx[ly][lx], yy[ly][lx], zz[ly][lx];

    char filename[64];

    for (BLASINT st = 0; st < ly*lx; st++)
    {	    
        int y = st/lx;
        int x = st%lx;

        double Sz[2][2] = {0.5,  0.0,
                           0.0, -0.5};

        double Sx[2][2] = {0.0,  0.5,
                           0.5,  0.0};

	double RSy[2][2] = {0.0, -0.5,
	                    0.5,  0.0};

	torch::Tensor SSZ = torch::rand({2, 2}, Crb[0].dtype());  // torch::tensor({0.5, 0.0, 0.0,-0.5}, Crb[0].dtype()).view({2, 2});
	torch::Tensor SSX = torch::rand({2, 2}, Crb[0].dtype());
        torch::Tensor SSY = torch::rand({2, 2}, Crb[0].dtype());

        for (int i = 0; i < 2; i++)
	{
	    for (int j = 0; j < 2; j++) 
	    {
		SSZ[i][j] = Sz[i][j];   
	        SSX[i][j] = Sx[i][j];
	        SSY[i][j] = RSy[i][j];	
	    }	    
	}

        auto sen = cross(Clb[st], Crb[st], Clt[st], Crt[st], El[st], Eb[st], Er[st], Et[st], bt[st]);
        auto senx = WTWcross(Clb[st], Crb[st], Clt[st], Crt[st], El[st], Eb[st], Er[st], Et[st], wf[y*lx+x], SSX);
        auto senz = WTWcross(Clb[st], Crb[st], Clt[st], Crt[st], El[st], Eb[st], Er[st], Et[st], wf[y*lx+x], SSZ);
        auto seny = WTWcross(Clb[st], Crb[st], Clt[st], Crt[st], El[st], Eb[st], Er[st], Et[st], wf[y*lx+x], SSY);

        cout.precision(10);
        cout.width(15);

	if (sen.is_complex())
	{
            cout << "Three components are : " << torch::real(senx/sen) << " + i"<<torch::imag(senx/sen) << " ;  " << torch::real(seny/sen) << " + i" <<torch::imag(seny/sen) << " ; " <<  torch::real(senz/sen) << " + i"<<torch::imag(senz/sen); 
            cout << sqrt(senx * senx + senz * senz - seny * seny)/sen << std::endl;
        }
	else
	{
	    cout << "Three components are : " << (senx/sen).item() << "   " << (seny/sen).item() << "   " << (senz/sen).item() << std::endl;
            cout << (sqrt(senx*senx + senz*senz - seny*seny)/sen).item() << std::endl;
	}

	xx[y][x] = senx/sen;
	yy[y][x] = seny/sen;
	zz[y][x] = senz/sen;
    }

    for (int y = 0; y < ly; y++)
    {
	for (int x = 0; x < lx; x++)    
	{
            int xp = (x+1)%lx;
            int yp = (y+1)%ly;

	    double sss; 
	    
	    if (xx[0][0].is_complex()) sss = torch::real((xx[y][x]*xx[y][xp]-yy[y][x]*yy[y][xp]+zz[y][x]*zz[y][xp])/sqrt(xx[y][x]*xx[y][x]-yy[y][x]*yy[y][x]+zz[y][x]*zz[y][x])/sqrt(xx[y][xp]*xx[y][xp]-yy[y][xp]*yy[y][xp]+zz[y][xp]*zz[y][xp])).item().toFloat();	
	    else sss = ((xx[y][x]*xx[y][xp]-yy[y][x]*yy[y][xp]+zz[y][x]*zz[y][xp])/sqrt(xx[y][x]*xx[y][x]-yy[y][x]*yy[y][x]+zz[y][x]*zz[y][x])/sqrt(xx[y][xp]*xx[y][xp]-yy[y][xp]*yy[y][xp]+zz[y][xp]*zz[y][xp])).item().toFloat();
	    
	    std::cout << "Angle = " << acos(sss)*180/3.14159265 << std::endl;
	}
    } 
}

torch::Tensor TensorModel::bondExpt()
{
    torch::Tensor hST;  

    setBondOperator(hST, bt[0]);
    
    torch::Tensor EnergyX[lenY*lenX], EnergyY[lenY*lenX];

    for (int y = 0; y < lenY; y++)
    {
        for (int x = 0; x < lenX; x++) EnergyX[y*lenX+x] = bondExptX(y, x, hST);
    }

    for (int y = 0; y < lenY; y++)
    {
	for (int x = 0; x < lenX; x++) EnergyY[y*lenX+x] = bondExptY(y, x, hST);
    }

    for (int st = 0 ; st < lenY*lenX; st++) cout << "Energy at (" << st/lenX <<", "<< st%lenX << ") is " << EnergyX[st].item() << "   " << EnergyY[st].item() << std::endl;

    std::cout << std::endl;

    torch::Tensor sumEX = torch::zeros({1}, EnergyX[0].dtype());
    torch::Tensor sumEY = torch::zeros({1}, EnergyY[0].dtype());

    for (int st = 0; st < lenY*lenX; st++) 
    {
	sumEX += EnergyX[st];
        sumEY += EnergyY[st];	
    }

//    std::cout << "X energy is " << (sumEX/(lenX*lenY)).item() << ", Y energy is " << (sumEY/(lenX*lenY)).item() << ", average energy is  "<< ((sumEX+sumEY)/(lenX*lenY*2)).item() << std::endl << std::endl;


    torch::Tensor EnergyCBT[lenY*lenX], EnergyCTB[lenY*lenX]; 

    for (BLASINT y = 0; y < lenY; y++)
    {
        for (BLASINT x = 0; x < lenX; x++)
        {
            EnergyCTB[y*lenX+x] = bondExptCTB(y, x, hST);
            EnergyCBT[y*lenX+x] = bondExptCBT(y, x, hST);   
        }
    }

    torch::Tensor sumEtb = torch::zeros({1}, EnergyCTB[0].dtype());
    torch::Tensor sumEbt = torch::zeros({1}, EnergyCBT[0].dtype());

    for (int st = 0; st < lenY*lenX; st++)
    {
	sumEtb += EnergyCTB[st];
        sumEbt += EnergyCBT[st];	
    }

    std::cout <<"Cross energy is " << (sumEtb/(lenY*lenX)).item() <<"   "<< (sumEbt/(lenY*lenX)).item() << std::endl;

    std::cout << "End of bond expectation!" << std::endl;

    return sumEX+sumEY + 0.2*(sumEtb+sumEbt);;
}

torch::Tensor TensorModel::bondExptX(const BLASINT &y, const BLASINT &x, torch::Tensor &hST)
{
    int xp = (x+1)%lenX;
    int yp = (y+1)%lenY;

    int yx = y * lenX + x;
    int yxp = y * lenX + xp;
    int ypx = yp * lenX + x;
    int ypxp = yp * lenX + xp;

/*First we calculate the energy of the X bond*/

    auto [WUWyx, WVWyxp] = gateDecomposition(wf[yx], wf[yxp], hST); 

    auto dimS = WUWyx.size(0);
    auto dima = WUWyx.size(1);
    auto dimb = WUWyx.size(2);
    auto dimc = WUWyx.size(3);
    auto dimd = WUWyx.size(4);

    auto vWUW = WUWyx.permute({1, 2, 0, 3, 4}).contiguous().view({dima, dimb, dimS*dimc, dimd});

    dima = WVWyxp.size(1);
    dimb = WVWyxp.size(2);
    dimc = WVWyxp.size(3);
    dimd = WVWyxp.size(4);
      
    auto vWVW = WVWyxp.view({dimS*dima, dimb, dimc, dimd});
/*
    torch::Tensor ltc = LTContraction(Clt[ypx], El[ypx], Et[ypx], bt[ypx]);
    torch::Tensor rtc = RTContraction(Crt[ypxp], Er[ypxp], Et[ypxp], bt[ypxp]);
    torch::Tensor lbc = LBContraction(Clb[yx], El[yx], Eb[yx], bt[yx]);
    torch::Tensor rbc = RBContraction(Crb[yxp], Er[yxp], Eb[yxp], bt[yxp]);                	
*/
    torch::Tensor rlt = torch::tensordot(RTC[ypxp], LTC[ypx], {1}, {0});
    torch::Tensor lrb = torch::tensordot(LBC[yx], RBC[yxp], {1}, {0});
    torch::Tensor ttbb = torch::tensordot(rlt, lrb, {1, 0}, {0, 1});

    auto lbcT = LBContraction(Clb[yx], El[yx], Eb[yx], vWUW);
    auto rbcT = RBContraction(Crb[yxp], Er[yxp], Eb[yxp], vWVW);    

    auto lrbcT = torch::tensordot(lbcT, rbcT, {1}, {0});
    auto ttbbT = torch::tensordot(rlt, lrbcT, {1, 0}, {0, 1});    

    return ttbbT/ttbb;
}

torch::Tensor TensorModel::bondExptY(const BLASINT &y, const BLASINT &x, torch::Tensor &hST)
{
    int xp = (x+1)%lenX;
    int yp = (y+1)%lenY;

    int yx = y * lenX + x;
    int yxp = y * lenX + xp;
    int ypx = yp * lenX + x;
    int ypxp = yp * lenX + xp;

/*Energy along the Y bond*/

    auto [WUWyx, WVWypx] = gateDecomposition(wf[yx], wf[ypx], hST);

    auto dimS = WUWyx.size(0);
    auto dima = WUWyx.size(1);
    auto dimb = WUWyx.size(2);
    auto dimc = WUWyx.size(3);
    auto dimd = WUWyx.size(4);	  
    
    auto vWUW = WUWyx.permute({1, 2, 3, 0, 4}).contiguous().view({dima, dimb, dimc, dimS*dimd});

    dima = WVWypx.size(1);
    dimb = WVWypx.size(2);
    dimc = WVWypx.size(3);
    dimd = WVWypx.size(4);

    auto vWVW = WVWypx.permute({1, 0, 2, 3, 4}).contiguous().view({dima, dimS*dimb, dimc, dimd});
/*	    
    torch::Tensor ltc = LTContraction(Clt[ypx], El[ypx], Et[ypx], bt[ypx]);
    torch::Tensor rtc = RTContraction(Crt[ypxp], Er[ypxp], Et[ypxp], bt[ypxp]);
    torch::Tensor lbc = LBContraction(Clb[yx], El[yx], Eb[yx], bt[yx]);
    torch::Tensor rbc = RBContraction(Crb[yxp], Er[yxp], Eb[yxp], bt[yxp]);   
*/	    
    torch::Tensor ltb = torch::tensordot(LTC[ypx], LBC[yx], {1}, {0}); 
    torch::Tensor rbt = torch::tensordot(RBC[yxp], RTC[ypxp], {1}, {0});	    
    torch::Tensor ttbb = torch::tensordot(ltb, rbt, {1, 0}, {0, 1});

    torch::Tensor lbcT = LBContraction(Clb[yx], El[yx], Eb[yx], vWUW);
    torch::Tensor ltcT = LTContraction(Clt[ypx], El[ypx], Et[ypx], vWVW);

    torch::Tensor ltbT = torch::tensordot(ltcT, lbcT, {1}, {0}); 
    torch::Tensor ttbbT = torch::tensordot(rbt, ltbT, {1, 0}, {0, 1}); 

    return ttbbT/ttbb;
}

torch::Tensor TensorModel::bondExptCTB(const BLASINT &y, const BLASINT &x, torch::Tensor &hST)
{
    int xp = (x+1)%lenX;
    int yp = (y+1)%lenY;

    int yx = y * lenX + x;
    int yxp = y * lenX + xp;
    int ypx = yp * lenX + x;
    int ypxp = yp * lenX + xp;

/*Energy along the Y bond*/

    auto [WUWypx, WVWyxp] = gateDecomposition(wf[ypx], wf[yxp], hST);

    auto dimS = WUWypx.size(0);
    auto dima = WUWypx.size(1);
    auto dimb = WUWypx.size(2);
    auto dimc = WUWypx.size(3);
    auto dimd = WUWypx.size(4);

    auto vWUW = WUWypx.permute({1, 2, 0, 3, 4}).contiguous().view({dima, dimb*dimS, dimc, dimd});

    dima = WVWyxp.size(1);
    dimb = WVWyxp.size(2);
    dimc = WVWyxp.size(3);
    dimd = WVWyxp.size(4);

    auto vWVW = WVWyxp.permute({1, 0, 2, 3, 4}).contiguous().view({dima*dimS, dimb, dimc, dimd});
/*
    torch::Tensor ltc = LTContraction(Clt[ypx], El[ypx], Et[ypx], bt[ypx]);
    torch::Tensor rtc = RTContraction(Crt[ypxp], Er[ypxp], Et[ypxp], bt[ypxp]);
    torch::Tensor lbc = LBContraction(Clb[yx], El[yx], Eb[yx], bt[yx]);
    torch::Tensor rbc = RBContraction(Crb[yxp], Er[yxp], Eb[yxp], bt[yxp]);
*/
    torch::Tensor ltb = torch::tensordot(LTC[ypx], LBC[yx], {1}, {0});
    torch::Tensor rbt = torch::tensordot(RBC[yxp], RTC[ypxp], {1}, {0});
    torch::Tensor ttbb = torch::tensordot(ltb, rbt, {1, 0}, {0, 1});

    torch::Tensor ltcT = LTContraction(Clt[ypx], El[ypx], Et[ypx], vWUW);
    torch::Tensor lrt  = torch::tensordot(RTC[ypxp], ltcT, {1}, {0});
    torch::Tensor lrtt = lrt.reshape({RTC[ypxp].size(0), LBC[yx].size(0), dimS});

    torch::Tensor rbcT = RBContraction(Crb[yxp], Er[yxp], Eb[yxp], vWVW);
    torch::Tensor rbcT2 = rbcT.reshape({LBC[yx].size(1), dimS, rbcT.size(1)});
    torch::Tensor lrb = torch::tensordot(LBC[yx], rbcT2, {1}, {0});

    torch::Tensor ttbbT = torch::tensordot(lrtt, lrb, {0, 1, 2}, {2, 0, 1});

    return ttbbT/ttbb;
}

/*cross from bottom(left) to top(right)*/

torch::Tensor TensorModel::bondExptCBT(const BLASINT &y, const BLASINT &x, torch::Tensor &hST)
{
    int xp = (x+1)%lenX;
    int yp = (y+1)%lenY;

    int yx = y * lenX + x;
    int yxp = y * lenX + xp;
    int ypx = yp * lenX + x;
    int ypxp = yp * lenX + xp;

    auto [WUWyx, WVWypxp] = gateDecomposition(wf[yx], wf[ypxp], hST);

    auto dimS = WUWyx.size(0);
    auto dima = WUWyx.size(1);
    auto dimb = WUWyx.size(2);
    auto dimc = WUWyx.size(3);
    auto dimd = WUWyx.size(4);

    auto vWUW = WUWyx.permute({1, 2, 3, 0, 4}).contiguous().view({dima, dimb, dimc*dimS, dimd});

    dima = WVWypxp.size(1);
    dimb = WVWypxp.size(2);
    dimc = WVWypxp.size(3);
    dimd = WVWypxp.size(4);

    auto vWVW = WVWypxp.view({dimS*dima, dimb, dimc, dimd});
/*
    torch::Tensor ltc = LTContraction(Clt[ypx], El[ypx], Et[ypx], bt[ypx]);
    torch::Tensor rtc = RTContraction(Crt[ypxp], Er[ypxp], Et[ypxp], bt[ypxp]);
    torch::Tensor lbc = LBContraction(Clb[yx], El[yx], Eb[yx], bt[yx]);
    torch::Tensor rbc = RBContraction(Crb[yxp], Er[yxp], Eb[yxp], bt[yxp]);
*/
    torch::Tensor ltb = torch::tensordot(LTC[ypx], LBC[yx], {1}, {0});
    torch::Tensor rbt = torch::tensordot(RBC[yxp], RTC[ypxp], {1}, {0});
    torch::Tensor ttbb = torch::tensordot(ltb, rbt, {1, 0}, {0, 1});  // denominator

    torch::Tensor lbcT = LBContraction(Clb[yx], El[yx], Eb[yx], vWUW); 
    torch::Tensor ltbt  = torch::tensordot(LTC[ypx], lbcT, {1}, {0});
    auto lbtt = ltbt.reshape({LTC[ypx].size(0), LBC[yx].size(1), dimS}); 

    torch::Tensor rtcT = RTContraction(Crt[ypxp], Er[ypxp], Et[ypxp], vWVW);
    torch::Tensor rbtT  = torch::tensordot(RBC[yxp], rtcT, {1}, {0});
    auto rbtt = rbtT.reshape({RBC[yxp].size(0), dimS, RTC[ypxp].size(1)}); 

    torch::Tensor ttbbT = torch::tensordot(lbtt, rbtt, {0, 1, 2}, {2, 0, 1});

    return ttbbT/ttbb;
}

void TensorModel::calculateECE()
{
    for (int y = 0; y < lenY; y++)
    {
	for (int x = 0; x < lenX; x++)   
	{
	    int yx = y * lenX + x;

	    LTC[yx] = LTContraction(Clt[yx], El[yx], Et[yx], bt[yx]);
            RTC[yx] = RTContraction(Crt[yx], Er[yx], Et[yx], bt[yx]);
            LBC[yx] = LBContraction(Clb[yx], El[yx], Eb[yx], bt[yx]);
            RBC[yx] = RBContraction(Crb[yx], Er[yx], Eb[yx], bt[yx]);
	}	
    }
}

void TensorModel::setBondOperator(torch::Tensor &bondopt, const torch::Tensor &topt)
{
/*wt is only used to obtain the dtype*/	

    bondopt = torch::tensor({{0.25, 0.00, 0.00, 0.00}, 
                             {0.00,-0.25, 0.50, 0.00}, 
                             {0.00, 0.50,-0.25, 0.00}, 
                             {0.00, 0.00, 0.00, 0.25}}, topt.dtype()).view({4, 4});
}

#endif
