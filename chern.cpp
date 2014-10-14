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

void cChern::distribution(){
  construction();
  int rank, size, recvcount, sendcount, stride;
  int *sendbuf, *recvbuf;
  int *sendcounts, *displs, *recvcounts, *displs_r, *recvcountsE, *displs_rE; 
  complex<double> chern_rank;
  double chern_rank_real, total_chern;
  double *curvature_rank, *curvature;
  double *bdgE_rank, *bdgE;
  const int root = 0;
  int offset, offsetE;
  Init(_argc, _argv);
  rank = COMM_WORLD.Get_rank();
  size = COMM_WORLD.Get_size();
  if (rank == root){ // send process is only root significant                   
    sendbuf = new int[_NKX2];
    for(int i = 0; i< _NKX2; ++i){
      sendbuf[i] = i;
    }
    sendcounts = new int[size];
    displs = new int[size];
    for(int i=0; i<size; i++){
      sendcounts[i] = compute_count(i,size);
      displs[i] = i*int(_NKX2/size);
    }
  }
  recvcount = compute_count(rank,size); // This is a rank dependent variable.   
  recvbuf = new int[recvcount]; // So is this array: rank dependent size        
  MPI_Scatterv(sendbuf,sendcounts,displs,MPI_INT,recvbuf,recvcount,MPI_INT,root,COMM_WORLD);
  stride = pblock4*recvcount*_NMAX;
  for(int ig = 0; ig<size; ++ig) {
    if (ig ==rank){
      chern_rank = complex<double> (0.0,0.0);
      cout << "rank" << ig << "has started"<< endl;
      curvature_rank = new double[recvcount];
      for (int i=0; i<recvcount; ++i) {
	clock_t start = clock(); 
        update(recvbuf[i]);
	clock_t end = clock(); 
	if (rank==root) cout << "task " << recvbuf[i] <<"out of " << recvcount << "used " << double (end-start)/ (double) CLOCKS_PER_SEC  << endl; 
	chern_rank += _chern;
        curvature_rank[i] = _temp_curv;
      }
      cout << "rank " << rank << " has finished "<< recvcount << " tasks, " << endl;
    }
  }
  chern_rank_real = chern_rank.imag();
  for(int ig = 0; ig<size; ++ig) {
    if (ig ==rank){
      cout << "rank" << ig << "has chern number"<< chern_rank << endl;
    }
    MPI_Barrier(COMM_WORLD);
  }
  MPI_Reduce(&chern_rank_real, &total_chern, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
  if (root==rank) {
    cout << "Total Chern Number is: " << total_chern << endl;
    curvature = new double [_NKX2];
    recvcounts = new int[size];
    displs_r = new int[size];
    offset = 0;
    for(int ig = 0;ig<size;++ig){
      recvcounts[ig] = compute_count(ig,size);
      displs_r[ig] = offset;
      offset += recvcounts[ig];
    }
  }
  MPI_Gatherv(curvature_rank,recvcount,MPI_DOUBLE,curvature,recvcounts,displs_r,MPI_DOUBLE,root,COMM_WORLD);
  if (root==rank) {
    ofstream curv_output, akx, aky;
    curv_output.open("curvature.OUT");
    akx.open("AKX.OUT");
    aky.open("AKY.OUT");
    assert(curv_output.is_open());
    assert(akx.is_open());
    assert(aky.is_open());
    for(int nky = 0;nky <_NKX;++nky){
      for(int nkx = 0;nkx<_NKX;++nkx){
        curv_output << curvature[nkx+nky*_NKX] << '\t';
        akx << gauss_k[nkx] << '\t';
        aky << gauss_k[nky] << '\t';
      }
      curv_output << endl;
      akx << endl;
      aky << endl;
    }
    curv_output.close();
    akx.close();
    aky.close();
    delete []curvature;
    delete []sendbuf;
    delete []sendcounts;
    delete []displs;
    delete []recvcounts;
    delete []displs_r;
  }
  delete []recvbuf;
  delete []curvature_rank;
  Finalize();
}

void cChern::construction(){
  update(-1); // arbitrary null construction.
  cout << "construction completed" << endl;
  gauss_k = new double [_NKX];
  gauss_w_k = new double [_NKX];
  gauss_lgwt(_NKX,-kmax,kmax,gauss_k,gauss_w_k);
}

void cChern::update(int nk){
  if (nk == -1) {
    _bdg_H.setZero(); // This is done only once.
    // The off-diagonal coupling introduced from time-dependent order parameter should be computed only here.
    for (int i = 0; i < pblock; ++i) {
      for (int j = 0; j<i;++j){
	if (j == i-1){
	  for(int in = 0; in < _NMAX;++in){
	    _bdg_H(i*2*_NMAX+in*2,  j*2*_NMAX+in*2).real()   =  _Delta0/2.0;
	    _bdg_H(i*2*_NMAX+in*2+1,j*2*_NMAX+in*2+1).real() = -_Delta0/2.0;
	  }
	}
      }
    }
  } else {
    int nkx = nk % _NKX;
    int nky = int (nk/_NKX);
    int lowerbound = -999; 
    int upperbound = -999; // ridiculous negative flag 
    double kx = gauss_k[nkx];
    double ky = gauss_k[nky];
    SelfAdjointEigenSolver<MatrixXcd> ces;
    complex<double> u,a,b,v,up,ap,bp,vp, Theta1,Theta2, temp;
    complex<double> myI (0.0,1.0);
    update_kxky(kx,ky);
    ces.compute(_bdg_H);
    _bdg_E = ces.eigenvalues();
    _bdg_V = ces.eigenvectors();
    for(int ip = 0; ip < pblock;++ip){
      if (_bdg_E[ip]/(M_PI/_T) >= -1.0) {
        lowerbound = ip;
        break;
      }
    }
    for(int ip = pblock; ip < pblock4;++ip){
      if (_bdg_E[ip]/(M_PI/_T) >= 1.0) {
        upperbound = ip-1;
        break;
      }
    }
    if (lowerbound < 0 || upperbound < 0){
      _temp_curv = 0.0;
      _chern = complex<double> (0.0,0.0);
      //      cout << "no contribution is added" << endl;
    } else {
      //      cout  <<"lower bound = " << lowerbound << " upper bound = " << upperbound << ", and " << upperbound-lowerbound+1 << " is considered for computation." <<endl;
      _chern = complex<double> (0.0,0.0);
      for(int ih = lowerbound; ih < pblock; ++ih) { // hole branch 
	for(int ip = pblock;ip<=upperbound;++ip){ // particle branch
	  Theta1 = complex<double> (0.0,0.0);
	  Theta2 = complex<double> (0.0,0.0);
	  for(int i = 0; i < pblock; ++i){ // frequency block adds up         
	    u = _bdg_V(i*2,ih);
	    v = _bdg_V(i*2+1,ih);
	    up = _bdg_V(i*2,ip);
	    vp = _bdg_V(i*2+1,ip);
	    Theta1 += -(2.0*_b*sin(kx)+_J*sin(kx)*cos(ky))*up*conj(u)
	      +_a*cos(kx)*vp*conj(u)
	      +_a*cos(kx)*up*conj(v)
	      +(2.0*_b*sin(kx)+_J*sin(kx)*cos(ky))*vp*conj(v);
	    Theta2 += -(2.0*_b*sin(ky)+_J*cos(kx)*sin(ky))*u*conj(up)
	      -myI*_a*cos(ky)*v*conj(up)
	      +myI*_a*cos(ky)*u*conj(vp)
	      +(2.0*_b*sin(ky)+_J*cos(kx)*sin(ky))*v*conj(vp);
	  }
	  _chern += -2.0*Theta1*Theta2/pow(_bdg_E[ih]-_bdg_E[ip],2.0)/(2.0*M_PI);
	}
      }
      _temp_curv = _chern.imag();
      _chern = _chern*gauss_w_k[nky]*gauss_w_k[nkx];
    }
  }
}


void cChern::update_kxky(double kx, double ky){
  int p,n;
  for (int i = 0; i < pblock; ++i) {
    p = i-_PMAX;
    for(int in = 0; in < _NMAX;++in){
      n = in + 1;
      // only lower left part of the matrix is needed for self-adjoint matrix storage.
      _bdg_H(i*2*_NMAX+in*2,i*2*_NMAX+in*2).real()     =  _mu-_J
	-2.0*_b*(2.0-cos(kx)-cos(ky))
	+_J*cos(kx)*cos(ky)
	+p*_omega;
      _bdg_H(i*2*_NMAX+in*2+1,i*2*_NMAX+in*2)   = complex<double>(_a*sin(kx),_a*sin(ky));
      _bdg_H(i*2*_NMAX+in*2+1,i*2*_NMAX+in*2+1).real() = -_mu+_J
	+2.0*_b*(2.0-cos(kx)-cos(ky))
	-_J*cos(kx)*cos(ky)
	+p*_omega;
    }
  }
}
