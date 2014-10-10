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
  double  mu, J, b, a, Delta0, omega, L;
  mu = 1.0;
  J = 1.5*mu;
  b = 1.5*mu;
  Delta0 = 1.0*mu;
  a = 4.0*mu;
  //omega = Delta0/0.07;
  omega = mu/0.07;
  L = 200;
  sPara para; para.mu = mu; para.J = J;para.b = b;para.a = a;para.Delta0=Delta0;para.omega = omega;para.L = L;
  cChern Chern(para, argc, argv);
  Chern.distribution();
  return 0;
}


