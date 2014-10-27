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
  hf = 1.5;
  muInf = 0.5;
  Tperiod = 0;
  sPara para; para.t = 0.2; para.h = hf;para.v = 1.0;
  sPhys phys; phys.mu = muInf; phys.T = Tperiod;
  cChern Chern(para, phys, argc, argv);
  Chern.construction();
  Chern.distribution();
  return 0;
}


