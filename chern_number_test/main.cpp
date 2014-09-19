//
//  main.cpp
//  Chern Number calculation of the Floquet state
//
//  Created by Dong Lin on 9/2/14.
//  Copyright (c) 2014 Dong Lin. All rights reserved.
//

#include "stdcpp.h"
#include "chern.h"
int main(int argc, char** argv){
	double  muInf, Tperiod;
	// TODO: modify dataset
	// dataset: (hi, hf) = (2.1, 0.9)
	ofstream bdg_output;
	bdg_output.open("minEg_fine.OUT");
	assert(bdg_output.is_open());
	for (double hf = 0.9;hf<1.21;hf=hf+0.001){
	muInf = 0.119725329786196;
	Tperiod = 26.239999999999998;
	sPara para; para.t = 0.2; para.h = hf;para.v = 1.2;
	sPhys phys; phys.mu = muInf; phys.T = Tperiod;
	cChern Chern(para, phys, argc, argv);
	Chern.construction();
	Chern.distribution();
        bdg_output << Chern.gap() << endl;
	}
	bdg_output.close();
	return 0;
}


