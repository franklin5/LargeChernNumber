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
    //    MPI_Barrier(COMM_WORLD);
  }
  chern_rank_real = chern_rank.real();
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
}

void cChern::update(int nk){
  if (nk == -1) {
    _bdg_H.setZero(); // This is done only once.
    _bdg_H(2,1) = -1.0;
    _bdg_H(3,0) =  1.0;

  } else {
    int nkx = nk % _NKX;     // --> the modulo (because nk = nkx+ nky * NKX )
    int nky = int (nk/_NKX); // --> the floor
    double kmax = 10.0; // TODO: modify momentum space cutoff value
    double kx = -kmax + nkx * kmax *2.0 /(_NKX-1);
    double ky = -kmax + nky * kmax *2.0 /(_NKX-1);
    SelfAdjointEigenSolver<MatrixXcd> ces;
    double dk = 0.5 * kmax * 2.0 /(_NKX-1);
    double qx, qy; 
    for (int i = 1; i < 5; ++i) {
      switch (i) {
      case 1:
	qx = kx - dk;
	qy = ky - dk;
	update_kxky(qx,qy);
	ces.compute(_bdg_H); 
	_loopA = ces.eigenvectors();
	//	cout << "_loopA is finished." << endl;
	break;
      case 2:
	qx = kx + dk;
	qy = ky - dk;
	update_kxky(qx,qy);
	ces.compute(_bdg_H); 
	_loopB = ces.eigenvectors();
	//	cout << "_loopB is finished." << endl;
	break;
      case 3:
	qx = kx + dk;
	qy = ky + dk;
	update_kxky(qx,qy);
	ces.compute(_bdg_H); 
	_loopC = ces.eigenvectors();
	//	cout << "_loopC is finished." << endl;
	break;
      case 4:
	qx = kx - dk;
	qy = ky + dk;
	update_kxky(qx,qy);
	ces.compute(_bdg_H); 
	_loopD = ces.eigenvectors();
	//	cout << "_loopD is finished." << endl;
	break;
      default:
	break;
      }
    }
    _chern = complex<double> (1.0,0.0);
    complex<double> temp;
    for(int ip = 0; ip < 2; ++ip) {
      temp = _loopA.col(ip).adjoint() * _loopB.col(ip);
      _chern = _chern * temp/abs(temp);
      temp = _loopB.col(ip).adjoint() * _loopC.col(ip);
      _chern = _chern * temp/abs(temp); 
      temp = _loopC.col(ip).adjoint() * _loopD.col(ip);
      _chern = _chern * temp/abs(temp); 
      temp = _loopD.col(ip).adjoint() * _loopA.col(ip);
      _chern = _chern * temp/abs(temp); 
    }
    _chern = complex<double>(log(std::abs(_chern)),atan(_chern.imag()/_chern.real()));
    _chern = _chern/2/M_PI*complex<double>(0.0,1.0);
    //    cout << _chern << endl;
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
