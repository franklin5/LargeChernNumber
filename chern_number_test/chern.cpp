/*
 * edge.cpp
 *
 *  Created on: Sep 2, 2014
 *      Author: Lin Dong
 */
#include "chern.h"

int cChern::compute_count(int rank, int size){
  // compute distribution count: the last rank does the remaineder job while the rest do the most even work.                                            
  int result;
  if (rank != size-1) {
    result = int(_NKX2/size);
  } else {
    result = int(_NKX2/size) + _NKX2 % size;
 }
  return result;
}

double cChern::gap(){
  return _bdg_E[pblock4/2];
}

void cChern::distribution(){
	update(0);
	SelfAdjointEigenSolver<MatrixXcd> ces;
	ces.compute(_bdg_H,0); // eigenvectors are also computed.
	for(int j = 0; j < pblock4; ++j){
	  _bdg_E[j]=ces.eigenvalues()[j];
	}
      
}
void cChern::construction(){
	update(-1); // arbitrary null construction.
}

void cChern::update(int nk){
  if (nk == -1) {
	  _bdg_H.setZero(); // This is done only once.
	  int p, q;
	  // The off-diagonal coupling introduced from time-dependent order parameter should be computed only here.
	  complex<double> Gamma2;
	  FILE *sf_inputR, *sf_inputI;
	  sf_inputR = fopen ("Rdata_2109.dat","r");
	  sf_inputI = fopen ("Idata_2109.dat","r");
	  assert (sf_inputR != NULL);
	  assert (sf_inputI != NULL);
	  double dt = 0.0005;
	  double t;
	  VectorXcd Delta_t(100000);
	  Delta_t.setZero();
	  int count = 0;
	  double reD, imD;
//	  ofstream test_output;
//	  test_output.open("test_Delta.OUT"); // TODO: modify output file name
//	  assert(test_output.is_open());
	  while (fscanf(sf_inputR, "%lf", &reD) != EOF && fscanf(sf_inputI, "%lf", &imD) != EOF ){
		Delta_t(count) = complex<double>(reD,imD);
//		test_output << reD << '\t' << imD << endl;
	  	count++;
	  }
//	  test_output.close();
	  fclose (sf_inputR);
	  fclose (sf_inputI);
	  for (int i = 0; i < pblock; ++i) {
	  		p = i-_PMAX;
	  		for (int j = 0; j < pblock; ++j) {
	  			q = j-_PMAX;
	  			Gamma2 = complex<double> (0.0,0.0);
	  			t = 0.0;
	  			for (int ig = 0; ig < count; ++ig) {
					Gamma2 +=  abs(Delta_t(ig)) *
							complex<double> (cos(2*M_PI*(q-p)*t/_T),-sin(2*M_PI*(q-p)*t/_T));
					t += dt;
				}
	  			Gamma2 = Gamma2/_T*dt;
//	  			cout << Gamma2 << endl;
	  			_bdg_H(i+2*pblock,j+pblock) = Gamma2;
	  			_bdg_H(i+3*pblock,j) = -Gamma2;
	  		}
	  }
  } else {
    double kx = 0;
    double ky = 0;
	
	  update_kxky(kx,ky);
  }
}

void cChern::update_kxky(double kx, double ky){
	double xi = kx*kx + ky*ky - _mu;
	int p;
	for (int i = 0; i < pblock; ++i) {
		p = i-_PMAX;
		_bdg_H(i,i) 	   = complex<double>(xi+_h+2*M_PI*p/_T,0.0);
		_bdg_H(i+pblock,i)   = complex<double>(_v*kx,_v*ky);
		_bdg_H(i+pblock,i+pblock)= complex<double>(xi-_h+2*M_PI*p/_T,0.0);
		_bdg_H(i+2*pblock,i+2*pblock) = complex<double>(-(xi+_h+2*M_PI*p/_T),0.0);
		_bdg_H(i+3*pblock,i+2*pblock) = complex<double>(_v*kx,-_v*ky);
		_bdg_H(i+3*pblock,i+3*pblock) = complex<double>(-(xi-_h+2*M_PI*p/_T),0.0);
	}
}
