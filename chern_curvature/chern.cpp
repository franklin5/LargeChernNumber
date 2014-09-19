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
  int rank, size, recvcount, sendcount, stride;
  int *sendbuf, *recvbuf;
  int *sendcounts, *displs, *recvcounts, *displs_r;
  double  *localEig, *TotalEig;
  complex<double> chern_rank;
  double chern_rank_real, total_chern;
  const int root = 0;
  int offset;
  //  SelfAdjointEigenSolver<MatrixXcd> ces;
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
  stride = pblock4*recvcount;
  localEig = new double[stride];  
  for(int ig = 0; ig<size; ++ig) {
    if (ig ==rank){
      chern_rank = complex<double> (0.0,0.0);
      for (int i=0; i<recvcount; ++i) {
	//cout << "rank = " << ig << "recvbuf[" << i << "] = " << recvbuf[i] << endl;
	update(recvbuf[i]);
	chern_rank += _chern;
      }
      cout << "rank " << rank << " has finished "<< recvcount << " tasks, " << " and chern_rank = " << chern_rank << endl;
    }
  }
  chern_rank_real = chern_rank.imag();
  MPI_Reduce(&chern_rank_real, &total_chern, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
  if (root==rank) {
    cout << "Total Chern Number is: " << total_chern << endl;
    delete []sendbuf;
    delete []sendcounts;
    delete []displs;
  }
  delete []recvbuf;
  Finalize();
}
void cChern::construction(){
  update(-1); // arbitrary null construction.
  gauss_k = new double [_NKX];
  gauss_w_k = new double [_NKX];
  gauss_lgwt(_NKX,-kmax,kmax,gauss_k,gauss_w_k);
}

void cChern::update(int nk){
  if (nk == -1) {
    _bdg_H.setZero(); // This is done only once.
    _bdg_H(2,1) =  1.0;
    _bdg_H(3,0) = -1.0;

  } else {
    int nkx = nk % _NKX;     // --> the modulo (because nk = nkx+ nky * NKX )
    int nky = int (nk/_NKX); // --> the floor
    //    double kx = -kmax + nkx * kmax *2.0 /(_NKX-1);
    //    double ky = -kmax + nky * kmax *2.0 /(_NKX-1);
    double kx = gauss_k[nkx];
    double ky = gauss_k[nky];
    SelfAdjointEigenSolver<MatrixXcd> ces;
    complex<double> myI(0.0,1.0);
    //    double dk = 0.5 * kmax * 2.0 /(_NKX-1);
    update_kxky(kx,ky);
    ces.compute(_bdg_H); 
    _bdg_E = ces.eigenvalues();
    _bdg_V = ces.eigenvectors();
    complex<double> u,a,b,v,up,ap,bp,vp,Theta1, Theta2;
    int i = 0;
    _chern = complex<double> (0.0,0.0);
    for (int ih = 0;ih<2;++ih){
      if ( _bdg_E(ih) > 0) cerr << "Error: hole energy is positive... Stop!" << endl;
      u = _bdg_V(i*4,ih);
      a = _bdg_V(i*4+1,ih);
      b = _bdg_V(i*4+2,ih);
      v = _bdg_V(i*4+3,ih);
      for(int ip = 2;ip<4;++ip){
	if ( _bdg_E(ip) < 0) cerr << "Error: particle energy is negative... Stop!" << endl;
	up = _bdg_V(i*4,ip);
	ap = _bdg_V(i*4+1,ip);
	bp = _bdg_V(i*4+2,ip);
	vp = _bdg_V(i*4+3,ip);
	Theta1 = 2*kx*up*conj(u)+_v*ap*conj(u)+_v*up*conj(a)+2*kx*ap*conj(a)-2*kx*bp*conj(b)+_v*vp*conj(b)+_v*bp*conj(v)-2*kx*vp*conj(v);
	Theta2 = 2*ky*conj(up)*u-myI*_v*conj(up)*a+myI*_v*conj(ap)*u+2*ky*conj(ap)*a-2*ky*conj(bp)*b+myI*_v*conj(bp)*v-myI*_v*conj(vp)*b-2*ky*conj(vp)*v;
	_chern += Theta1*Theta2/pow(_bdg_E[ih]-_bdg_E[ip],2.0);
      }
    }
    _chern = -2.0*_chern*gauss_w_k[nkx]*gauss_w_k[nky]/(2.0*M_PI);
  }
}    

void cChern::update_kxky(double kx, double ky){
  double xi = kx*kx + ky*ky - _mu;
  // only lower left part of the matrix is needed for storage.
    _bdg_H(0,0) = complex<double>(xi+_h,0.0);
    _bdg_H(1,0) = complex<double>(_v*kx,_v*ky);
    _bdg_H(1,1) = complex<double>(xi-_h,0.0);
    _bdg_H(2,2) = complex<double>(-(xi+_h),0.0);
    _bdg_H(3,2) = complex<double>(_v*kx,-_v*ky);
    _bdg_H(3,3) = complex<double>(-(xi-_h),0.0);
}
