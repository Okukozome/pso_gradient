#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <sys/resource.h>

#include "iPEPS.h"
#include "TensorModel.h"

iPEPS::iPEPS(std::vector<torch::Tensor> &gyx, const bool &ifgrad)
{
    for (int yx = 0; yx < LY*LX; yx++) Gyx[yx] = gyx[yx];	    

    char filename[128];

    if (ifgrad)
    {    
        for (int y = 0; y < LY; y++)
        {
            for(int x = 0; x < LX; x++)
            {
                stringname(64, filename, "Gyx", y, x);
                register_parameter(filename, Gyx[y*LX+x]);
            }
	}
    }

    J1 = 1.0;
    J2 = 0.57;
    delta = 0.1;

//    std::cout <<"J1 = " << J1 << ", J2 = " << J2 << ", delta = " << delta << std::endl;

/*correction to J1*/

    deltaX = 0.0;
    deltaY = 0.0;

//    std::cout <<"deltaX = " << deltaX <<"   deltaY = "<< deltaY << std::endl;

//    std::cout <<"LY = " << LX <<"   LX = "<< LY << std::endl; 

    prvIFC = true;
}

iPEPS::~iPEPS()
{
}

torch::Tensor iPEPS::groundEnergy(const int &fidx, const char &state, const BLASINT &chi, const std::pair<double, double> &epspr, const bool &ifdynamic, const bool &ifexpt, const bool &iftodisk, const bool &printable)
{
    char rst = state;
    if (!prvIFC) rst = 'I';

    TensorModel tm(fidx, LY, LX, Gyx, rst, chi);

//    std::cout <<"CTMRG chi is " << chi << "  id = "<< ifdynamic << std::endl;

    prvIFC = tm.getCTM(chi, ifdynamic, epspr, iftodisk, printable);
  
    tm.calculateECE();

    auto hST = torch::tensor({{0.25, 0.00, 0.00, 0.00}, 
                               {0.00,-0.25, 0.50, 0.00},  
                               {0.00, 0.50,-0.25, 0.00},  
                               {0.00, 0.00, 0.00, 0.25}}, Gyx[0].dtype()).view({4, 4});

    int lenY = tm.lenY;
    int lenX = tm.lenX;

    torch::Tensor EnergyX[lenY*lenX], EnergyY[lenY*lenX];

    for (int y = 0; y < lenY; y++)
    {
        for (int x = 0; x < lenX; x++) EnergyX[y*lenX+x] = tm.bondExptX(y, x, hST);
    }

    for (int y = 0; y < lenY; y++)
    {
        for (int x = 0; x < lenX; x++) EnergyY[y*lenX+x] = tm.bondExptY(y, x, hST);
    }

    torch::Tensor sumEX = torch::zeros({1}, EnergyX[0].dtype());
    torch::Tensor sumEY = torch::zeros({1}, EnergyY[0].dtype());

    if (printable) std::cout <<"YXEnergy :" << std::endl;

    for (int y = 0; y < LY; y++)
    {
	for (int x = 0; x < LX; x++)    
	{
	    int st = y*lenX+x;

	    double coeffY = y%2==0 ? J1*(1.0+deltaY) : J1;	
	    sumEY += coeffY*EnergyY[st];

	    double coeffX = x%2==1 ? J1*(1.0+deltaX) : J1;
	    sumEX += coeffX*EnergyX[st];

	    if (printable) std::cout<<"Energy at (" << y <<", "<< x << ") is " << EnergyY[st].item()<< "   "<< EnergyX[st].item() << "  :  "<<(coeffY*EnergyY[st]).item()<<"   "<<(coeffX*EnergyX[st]).item()<<std::endl;
	}
    }

    //std::cout << "X energy is " << (sumEX/(lenX*lenY)).item() << ", Y energy is " << (sumEY/(lenX*lenY)).item() << ", average energy is  "<< ((sumEX+sumEY)/(lenX*lenY*2)).item() << std::endl << std::endl;

/*cross terms*/

    torch::Tensor EnergyCBT[lenY*lenX], EnergyCTB[lenY*lenX];

    for (int y = 0; y < lenY; y++)
    {
        for (int x = 0; x < lenX; x++)
        {
            EnergyCTB[y*lenX+x] = tm.bondExptCTB(y, x, hST);
            EnergyCBT[y*lenX+x] = tm.bondExptCBT(y, x, hST);
        }
    }

    torch::Tensor sumEtb = torch::zeros({1}, EnergyCTB[0].dtype());
    torch::Tensor sumEbt = torch::zeros({1}, EnergyCBT[0].dtype());

    for (int y = 0; y < lenY; y++)
    {
	for (int x = 0; x <lenX; x++)    
	{
	    int st = y*lenX+x;

            double coeff = (x+y)%2==0 ? J2*(1+delta) : J2*(1-delta);

            sumEtb += coeff*EnergyCTB[st];	   
	    sumEbt += coeff*EnergyCBT[st];

//	    if (printable) std::cout << "Cross energy at (" << y <<", "<< x << ") is " << EnergyCTB[st].item() << "   " << EnergyCBT[st].item() << "  :   " << (coeff*EnergyCTB[st]).item() <<"  "<< (coeff*EnergyCBT[st]).item() << std::endl; 
	}
    }

//    std::cout <<"Cross energy is " << (sumEtb/(lenY*lenX)).item() <<"   "<< (sumEbt/(lenY*lenX)).item() << std::endl;

    auto sumE = (sumEX + sumEY) + (sumEtb + sumEbt);

    if (printable) std::cout << "Energy per site is " << (sumE/(lenY*lenX)).item() << std::endl;

    if (ifexpt) tm.siteExpt();

    return sumE; 
}

void iPEPS::savePEPS()
{
    char filename[128];

    for (int y = 0; y < LY; y++)
    {
	for (int x = 0; x < LX; x++)    
	{	
            stringname(64, filename, "Gyx", y, x);	    	
	    torch::save(Gyx[y*LX+x], filename);
	}
    }
}

std::tuple<double, double, double> iPEPS::memUsage(void)
{
    double vmk = 0.0;
    double rssk = 0.0;

    ifstream stat_stream("/proc/self/stat",ios_base::in); 
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string o, itrealvalue, starttime;
    long vsize;
    long rss;
    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice >> o >> itrealvalue >> starttime >> vsize >> rss;
    stat_stream.close();
    long page_size_kb = sysconf(_SC_PAGE_SIZE)/1024;
    vmk = vsize/1024.0/1024.0/1024.0;
    rssk = rss * page_size_kb/1024.0/1024.0;
    
    struct rusage  usg;

    getrusage(RUSAGE_SELF, &usg); 

    return std::tuple<double, double, double>(vmk, rssk, usg.ru_maxrss/1024.0/1024.0);    
}
