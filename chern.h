/*
 * edge.h
 *
 *  Created on: Sep 2, 2014
 *      Author: Lin Dong
 */
#ifndef EDGE_H_
#define EDGE_H_
#include "stdcpp.h"
#include "lgwt.h"
class cChern {
private:
  int _argc;
  char** _argv;
	double _J, _a, _b;
	double _mu, _Delta0, _omega, _L, _T;
	int _PMAX, pblock,pblock4, _MomentumSpaceCutoff, _NKX,_NKX2,_NMAX, _SMAX;
	VectorXd _bdg_E;
	MatrixXcd _bdg_V,_bdg_H;
	complex<double> _chern;
	double _temp_curv;
	double kmax;
	double* gauss_k, *gauss_w_k;
public:
	cChern(const sPara& para, int argc, char** argv)
	  :_argc(argc), _argv(argv),
	  _J(para.mu), _a(para.a), _b(para.b),
	  _mu(para.mu),_Delta0(para.Delta0),_omega(para.omega),_L(para.L),_T(2*M_PI/_omega),
	  // TODO: modify frequency cutoff
	  _PMAX(11), // number of frequency cutoff for time expansion
	  pblock(2*_PMAX+1),
	  // TODO: modify mmtn space cutoff for the bulk system
	  pblock4(2*pblock),
	  _NKX(101),
	  _NKX2(_NKX*_NKX),
	  _NMAX(1),
	  _SMAX(pblock4*_NMAX),
	  _bdg_E(pblock4*_NMAX),
	  _bdg_V(pblock4*_NMAX,pblock4*_NMAX),
	  _bdg_H(pblock4*_NMAX,pblock4*_NMAX),_chern(1.0,0.0),
	  _temp_curv(0.0),
	  kmax(M_PI){}
	  ~cChern(){
	    delete []gauss_k;
	    delete []gauss_w_k;}
	int compute_count(int,int);
	void distribution();
	void construction();
	void update(int);
	void update_kxky(double, double);
};
#endif /* EDGE_H_ */
