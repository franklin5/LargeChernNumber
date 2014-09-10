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
	double  muInf, Tperiod, hf;
	// TODO: modify dataset
	// dataset: (hi, hf) = (2.1, 0.9)
	hf = 0.9;
	muInf = 0.119725329786196;
	Tperiod = 26.239999999999998;
	sPara para; para.t = 0.2; para.h = hf;para.v = 1.2;
	sPhys phys; phys.mu = muInf; phys.T = Tperiod;
	cChern Chern(para, phys, argc, argv);
	Chern.distribution();
	return 0;
}


