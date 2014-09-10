/*
 * edge.h
 *
 *  Created on: Sep 2, 2014
 *      Author: Lin Dong
 */
#ifndef EDGE_H_
#define EDGE_H_
#include "stdcpp.h"
class cChern {
private:
  int _argc;
  char** _argv;
	double _Eb, _h, _v;
	double _mu, _T;
	int _PMAX, pblock,pblock4, _MomentumSpaceCutoff, _NKX, _NKX2;
	VectorXd _bdg_E;
	VectorXcd _bdg_u, _bdg_a, _bdg_b, _bdg_v;
	MatrixXcd _bdg_H;
public:
	cChern(const sPara& para, const sPhys& phys,  int argc, char** argv)
	  :_argc(argc), _argv(argv),
	  _Eb(para.t), _h(para.h), _v(para.v),
	  _mu(phys.mu),_T(phys.T),
	  // TODO: modify frequency cutoff
	  _PMAX(60), // number of frequency cutoff for time expansion
	  pblock(2*_PMAX+1),
	  // TODO: modify mmtn space cutoff for the bulk system
	  pblock4(4*pblock),
	  _MomentumSpaceCutoff(50),
	  _NKX(2*_MomentumSpaceCutoff+1),
	  _NKX2(_NKX*_NKX),
	  _bdg_E(pblock4),
	  _bdg_u(pblock), _bdg_a(pblock), _bdg_b(pblock), _bdg_v(pblock),
	  _bdg_H(pblock4,pblock4){}
	  int compute_count(int,int);
	void distribution();
	void construction();
	void update(int);
	void update_kxky(double, double);
};
#endif /* EDGE_H_ */
