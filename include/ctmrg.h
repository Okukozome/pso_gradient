#include <math.h>

#include "globalconfig.h"
#include "stringname.h"
#include "autograd.h"

#include "StepCTMRG.h"

/*
C  ---- 0    C  1 ----     C  0           C    1      E  ---- 0     E  2 ----    E   0  1  2    E  -------
   |                 |        |                |         |--- 1        1 ---|        |  |  |       |  |  |
   1                 0        ---- 1      0 ----         ---- 2        0 ----        -------       2  1  0

*/

#if !defined __CTMRG_H
#define __CTMRG_H

#define MAXCST (256)

class CTMRG
{
    public:
	const int fileidx;

        const int lenY, lenX;

	torch::Tensor* wf;  // wavefunction pointer in CTMRG

	std::vector<torch::Tensor> bt;

        std::vector<torch::Tensor> Clt, Clb, Crt, Crb;
        std::vector<torch::Tensor> El, Er, Et, Eb;	

        StepCTMRG cS;

    public:
        CTMRG(const int &fidx, const int &ylen, const int &xlen, torch::Tensor* wt, const char &state, const BLASINT &chi);

        bool getCTM(const BLASINT &chi, const bool &ifdynamic, const std::pair<double, double> &epspr, const bool &istodisk, const bool &printable);    
     
    public:
	virtual torch::Tensor getExpt() = 0;
};

CTMRG::CTMRG(const int &fidx, const int &ylen, const int &xlen, torch::Tensor* wt, const char &state, const BLASINT &chii) : fileidx(fidx), lenY(ylen), lenX(xlen), wf(wt)
{
    assert(MAXCST >= lenY*lenX);

    bt.resize(lenX*lenY);

    for (int y = 0; y < lenY; y++)
    {
        for (int x = 0; x < lenX; x++)
        {
            auto ww = torch::tensordot(torch::conj_physical(wf[y*lenX+x]), wf[y*lenX+x], {0}, {0}).permute({0, 4, 1, 5, 2, 6, 3, 7}).contiguous();
            bt[y*lenX+x] = ww.view({ww.size(0)*ww.size(1), ww.size(2)*ww.size(3), ww.size(4)*ww.size(5), ww.size(6)*ww.size(7)});
        }
    }

    for (int y = 0; y < lenY; y++)
    {
        int yp = (y+1)%lenY;
        int ym = (y-1+lenY)%lenY;

        for (int x = 0; x < lenX; x++)
        {
            int xp = (x+1)%lenX;
            int xm = (x-1+lenX)%lenX;

            int yx = y*lenX+x;
            int yxp = y*lenX+xp;
            int yxm = y*lenX+xm;
            int ypx = yp*lenX+x;
            int ymx = ym*lenX+x;

            assert(bt[yx].size(0) == bt[yxm].size(2));
            assert(bt[yx].size(1) == bt[ymx].size(3));
            assert(bt[yx].size(2) == bt[yxp].size(0));
            assert(bt[yx].size(3) == bt[ypx].size(1));
        }
    }

    assert(state == 'I' || state == 'i' || state == 'C' || state == 'c');

    Clb.resize(lenX*lenY);
    Crb.resize(lenX*lenY);
    Clt.resize(lenX*lenY);
    Crt.resize(lenX*lenY);

    El.resize(lenX*lenY);
    Er.resize(lenX*lenY);
    Et.resize(lenX*lenY);
    Eb.resize(lenX*lenY);

    if (state == 'I' || state == 'i')
    {
        // --- [日志] 初始状态，进行随机初始化 ---
        std::cout << "[Debug] Initializing tensors for fidx: " << fileidx << " in 'I' state (random)." << std::endl;
        for (int i = 0; i < lenX*lenY; i++)
        {
            Clb[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
            Crb[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
            Clt[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
            Crt[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
        }

        for (int y = 0; y < lenY; y++)
        {
            for (int x = 0; x < lenX; x++)
            {
                int yx = y*lenX+x;
                El[yx] = torch::rand({chii, bt[yx].size(0), chii}, bt[yx].dtype())-0.3;
                Er[yx] = torch::rand({chii, bt[yx].size(2), chii}, bt[yx].dtype())-0.3;
                Et[yx] = torch::rand({chii, bt[yx].size(3), chii}, bt[yx].dtype())-0.3;
                Eb[yx] = torch::rand({chii, bt[yx].size(1), chii}, bt[yx].dtype())-0.3;
            }
        }
    }
    else // (state == 'C' or 'c')
    {
        // --- [日志] 尝试从文件加载 ---
        std::cout << "[Debug] Attempting to load tensors for fidx: " << fileidx << " in 'C' state." << std::endl;
        try
        {
            char filename[128];
            for (int i = 0; i < lenX*lenY; i++)
            {
                stringname(80, filename, fileidx, "Clb", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(Clb[i], filename);

                stringname(80, filename, fileidx, "Crb", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(Crb[i], filename);

                stringname(80, filename, fileidx, "Clt", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(Clt[i], filename);

                stringname(80, filename, fileidx, "Crt", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(Crt[i], filename);

                stringname(80, filename, fileidx, "El", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(El[i], filename);

                stringname(80, filename, fileidx, "Er", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(Er[i], filename);

                stringname(80, filename, fileidx, "Et", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(Et[i], filename);

                stringname(80, filename, fileidx, "Eb", i);
                std::cout << "[Debug] Loading file: " << filename << std::endl;
                torch::load(Eb[i], filename);
            }
            // --- [日志] 加载成功 ---
            std::cout << "[Debug] All tensors for fidx: " << fileidx << " loaded successfully." << std::endl;
        }
        catch (const c10::Error& e)
        {
            // --- 修改开始: 添加 OpenMP critical 来同步输出 ---
            #pragma omp critical
            {
                // 将所有 std::cerr 和 std::cout 语句移到这个块内
                std::cerr << "[Debug] Caught exception for fidx: " << fileidx << ". Falling back to random init." << std::endl;
                std::cerr << "Error loading the tensor for fidx: " << fileidx << " - " << e.what() << "\n";
            }
            // --- 修改结束 ---

            // 随机化初始化的代码保持不变
            for (int i = 0; i < lenX*lenY; i++)
            {
                Clb[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
                Crb[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
                Clt[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
                Crt[i] = torch::rand({chii, chii}, bt[i].dtype())-0.2;
            }

            for (int y = 0; y < lenY; y++)
            {
                for (int x = 0; x < lenX; x++)
                {
                    int yx = y*lenX+x;
                    El[yx] = torch::rand({chii, bt[yx].size(0), chii}, bt[yx].dtype())-0.3;
                    Er[yx] = torch::rand({chii, bt[yx].size(2), chii}, bt[yx].dtype())-0.3;
                    Et[yx] = torch::rand({chii, bt[yx].size(3), chii}, bt[yx].dtype())-0.3;
                    Eb[yx] = torch::rand({chii, bt[yx].size(1), chii}, bt[yx].dtype())-0.3;
                }
            }
        }
    }
}

bool CTMRG::getCTM(const BLASINT &chi, const bool &ifdynamic, const std::pair<double, double> &epspr, const bool &iftodisk, const bool &printable)
{
    assert(chi > 0);

    char filename[128];

    double psprot[4] = {0.0, 0.0, 0.0, 0.0}, ptprot[4] = {0.0, 0.0, 0.0, 0.0};

    const BLASINT iternum = 30;

    bool ifconverged = false;

    auto timer_start = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds duration{0};

    BLASINT itn;
    for (itn = 0; itn < iternum; itn++)
    {
        if ((ifdynamic && !ifconverged) || !ifdynamic)	
	{
	    torch::NoGradGuard no_grad;
	    duration += cS.oneIteration(chi, ptprot, lenY, lenX, bt, Clt, Clb, Crt, Crb, El, Er, Et, Eb); 

            if (ifdynamic && itn%3 == 0)
            {
                ifconverged = true;

		double eps = epspr.first;
		double coeff = epspr.second;

                if (fabs(ptprot[0] - psprot[0]) > coeff*eps*ptprot[0] || fabs(ptprot[1] - psprot[1]) > coeff*eps*ptprot[1]) ifconverged = false;
                if (fabs(ptprot[2] - psprot[2]) > coeff*eps*ptprot[2] || fabs(ptprot[3] - psprot[3]) > coeff*eps*ptprot[3]) ifconverged = false;

                if (fabs(ptprot[0]-ptprot[1]) > eps*(ptprot[0]+ptprot[1]) || fabs(ptprot[2]-ptprot[3]) > eps*(ptprot[2]+ptprot[3])) ifconverged = false;

                psprot[0] = ptprot[0];
                psprot[1] = ptprot[1];
                psprot[2] = ptprot[2];
                psprot[3] = ptprot[3];
            }
	}

	if (ifconverged) break;
    }

    for (BLASINT n = 0; n < 4; n++) 
    {
	auto STbt = torch::cat(bt, 0);    
	auto STClt = torch::cat(Clt, 0);   
        auto STClb = torch::cat(Clb, 0);
	auto STCrt = torch::cat(Crt, 0);
	auto STCrb = torch::cat(Crb, 0);
	auto STEl = torch::cat(El, 0);
	auto STEr = torch::cat(Er, 0);
	auto STEt = torch::cat(Et, 0);
	auto STEb = torch::cat(Eb, 0);

	auto CE = TT(chi, lenY, lenX, STbt, STClt, STClb, STCrt, STCrb, STEl, STEr, STEt, STEb);    

	Clt = torch::split(CE[0], chi, 0);
	Clb = torch::split(CE[1], chi, 0);
	Crt = torch::split(CE[2], chi, 0);
	Crb = torch::split(CE[3], chi, 0);
        El = torch::split(CE[4], chi, 0);
	Er = torch::split(CE[5], chi, 0);
	Et = torch::split(CE[6], chi, 0);
	Eb = torch::split(CE[7], chi, 0);
    }

    if (iftodisk || ifconverged)
    {
        for (int i = 0; i < lenX*lenY; i++)
        {
            stringname(80, filename, fileidx, "Clb", i);
            torch::save(Clb[i], filename);	   

            stringname(80, filename, fileidx, "Clt", i);
            torch::save(Clt[i], filename);

            stringname(80, filename, fileidx, "Crb", i);
            torch::save(Crb[i], filename);
 
            stringname(80, filename, fileidx, "Crt", i);
            torch::save(Crt[i], filename);

            stringname(80, filename, fileidx, "El", i);
            torch::save(El[i], filename);

            stringname(80, filename, fileidx, "Er", i);
            torch::save(Er[i], filename);

            stringname(80, filename, fileidx, "Eb", i);
            torch::save(Eb[i], filename);

            stringname(80, filename, fileidx, "Et", i);
            torch::save(Et[i], filename);
        }
    }

    if (printable)
    {
        auto timer_ctm = std::chrono::high_resolution_clock::now();
        auto duration_ctm = duration_cast<std::chrono::milliseconds>(timer_ctm - timer_start);
        cout << "One CTMRG completed in " << duration_ctm.count() / 1000.0 << " seconds for " << itn << "/"<< iternum << " iterations." << endl;
    }

    return ifconverged;
}
#endif
