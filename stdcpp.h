//
//  stdcpp.h
//  Chern number calculation
//
//  Created by Dong Lin on 9/2/14.
//  Copyright (c) 2014 Dong Lin. All rights reserved.
//

#ifndef tBdG_stdcpp_h
#define tBdG_stdcpp_h
#include <mpi.h>
#include <iostream>
#include <new>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <Eigen/Eigenvalues>
#include <time.h>

using namespace std;
using namespace Eigen;
using namespace MPI;
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct sPhys {
    double mu;
    double T;
};

struct sPara {
    double t; // inverse scattering length, or bound state energy Eb in 2D
    double h; // zeeman field
    double v; // rashba SOC strength
};

struct sConf {
    double L; // length of confinement
    int NL; // number of grid
    double dl; // finite difference
};


#endif
