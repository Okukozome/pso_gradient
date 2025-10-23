#ifndef _IPEPS_H
#define _IPEPS_H

#include <torch/torch.h>
#include <iostream>


#include "globalconfig.h"
#include "localconfig.h"
#include "stringname.h"

class iPEPS : public torch::nn::Module
{
    public:
        torch::Tensor Gyx[LY*LX]; 

        double J1, J2, delta; 

        double deltaX, deltaY;   // adjust J1 to contruct a columnar or plaquette ground state

    public:
	iPEPS(std::vector<torch::Tensor> &gyx, const bool &ifgrad = false);

	virtual ~iPEPS();

        torch::Tensor groundEnergy(const int &fidx, const char &state, const BLASINT &chi, const std::pair<double, double> &epspr, const bool &ifdynamic = false, const bool &ifexpt = false, const bool &iftodisk = false, const bool &printable = false);

        void savePEPS();

	std::tuple<double, double, double> memUsage(void);

    private:
        bool prvIFC;    // if previous ctmrg is converged, it is true; else it is false.	
};

#endif
