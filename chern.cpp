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
  int *sendcounts, *displs, *recvcounts, *displs_r;
  double  *localEig, *TotalEig;
  complex<double> chern_rank;
  double chern_rank_real, total_chern;
  const int root = 0;
  int offset;
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
  for(int ig = 0; ig<size; ++ig) {
    if (ig ==rank){
      chern_rank = complex<double> (0.0,0.0);
      cout << "rank" << ig << "has started"<< endl;
      for (int i=0; i<recvcount; ++i) {
	clock_t start = clock(); 
        update(recvbuf[i]);
	clock_t end = clock(); 
	//	if (rank==root) cout << "task " << recvbuf[i] <<"out of " << recvcount << "used " << double (end-start)/ (double) CLOCKS_PER_SEC  << endl; 
        chern_rank += _chern;
      }
      cout << "rank " << rank << " has finished "<< recvcount << " tasks, " <<	" and chern_rank = " << chern_rank << endl;
    }
  }
  chern_rank_real = chern_rank.imag(); // curvature approach
  for(int ig = 0; ig<size; ++ig) {
    if (ig ==rank){
    cout << "rank" << ig << "has chern number"<< chern_rank << endl;
    }
    MPI_Barrier(COMM_WORLD);
  }
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
  cout << "construction completed" << endl;
  gauss_kx = new double [_NKX];gauss_w_kx = new double [_NKX];
  gauss_lgwt(_NKX,-kmax,kmax,gauss_kx,gauss_w_kx);
  gauss_ky = new double [_NKX];gauss_w_ky = new double [_NKX];
  gauss_lgwt(_NKX,-kmax,kmax,gauss_ky,gauss_w_ky);
}

void cChern::update(int nk){
  if (nk == -1) {
    _bdg_H.setZero(); // This is done only once.
    int p, q;
    // The off-diagonal coupling introduced from time-dependent order parameter should be computed only here.
    complex<double> Gamma2;
    FILE *sf_inputR, *sf_inputI;
    // TODO: modify input file name   
    sf_inputR = fopen ("Rdata_2109.dat","r"); 
    sf_inputI = fopen ("Idata_2109.dat","r");
    assert (sf_inputR != NULL);
    assert (sf_inputI != NULL);
    double dt = 0.0005;
    double t = 0.0;
    VectorXcd Delta_t(100000);
    Delta_t.setZero();
    int count = 0;
    double reD, imD;
    while (fscanf(sf_inputR, "%lf", &reD) != EOF && fscanf(sf_inputI, "%lf", &imD) != EOF ){
      Delta_t(count) = complex<double>(reD,imD);
      count++;
    }
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
	_bdg_H(i+2*pblock,j+pblock) = Gamma2;
	_bdg_H(i+3*pblock,j) = -Gamma2;
      }
    }
  } else {
    int nkx = nk % _NKX;     // --> the modulo (because nk = nkx+ nky * NKX )   
    int nky = int (nk/_NKX); // --> the floor                                   
    int lowerbound = -999; // negative flag                                     
    //    double kmax = 2.0; // TODO: modify momentum space cutoff value. If this is too large, say 5.0, the diagonalization result is strongly inaccurate...       
    //    double kx = -kmax + nkx * kmax *2.0 /(_NKX-1);
    //    double ky = -kmax + nky * kmax *2.0 /(_NKX-1);
    double kx = gauss_kx[nkx];
    double ky = gauss_ky[nky];
    SelfAdjointEigenSolver<MatrixXcd> ces;
    double dk = 0.5 * kmax * 2.0 /(_NKX-1);
    complex<double> u,a,b,v,up,ap,bp,vp, Theta1,Theta2, temp;
    complex<double> myI (0.0,1.0);
    update_kxky(kx,ky);
    ces.compute(_bdg_H);
    _loopA = ces.eigenvectors();
    for(int ip = 0; ip < 2*pblock;++ip){
      if (ces.eigenvalues()[ip]/(M_PI/_T) > -1) {
	lowerbound = ip;
	break;
      }
    }
    if (lowerbound < 0 || lowerbound > 2*pblock){
      _chern = complex<double> (0.0,0.0); // no contribution needs to be added
      cout << "no contribution is added" << endl;
    } else {
      cout  <<"lower bound = " << lowerbound << " upper bound = " << 2*pblock <<endl;
      _chern = complex<double> (0.0,0.0);
      Theta1 = complex<double> (0.0,0.0);
      Theta2 = complex<double> (0.0,0.0);
      temp = complex<double> (0.0,0.0);
 for(int ip = lowerbound; ip < 2*pblock; ++ip) {
   for(int iq = lowerbound;iq<2*pblock;++iq){
     if (iq != ip){
       //       cout << "ip=" << ip << "iq" << iq << endl;
       for(int i = 0;i<pblock;++i){
	 u = _loopA(i*4,ip);
	 a = _loopA(i*4+1,ip);
	 b = _loopA(i*4+2,ip);
 	 v = _loopA(i*4+3,ip);
	 up = _loopA(i*4,iq);
	 ap = _loopA(i*4+1,iq);
	 bp = _loopA(i*4+2,iq);
 	 vp = _loopA(i*4+3,iq);
	 Theta1 += 2*kx*up*conj(u)+_v*ap*conj(u)+_v*up*conj(a)+2*kx*ap*conj(a)-2*kx*bp*conj(b)+_v*vp*conj(b)+_v*bp*conj(v)-2*kx*vp*conj(v);
	 Theta2 += 2*ky*conj(up)*u-myI*_v*conj(up)*a+myI*_v*conj(ap)*u+2*ky*conj(ap)*a-2*ky*conj(bp)*b+myI*_v*conj(bp)*v-myI*_v*conj(vp)*b-2*ky*conj(vp)*v;
	 }
       temp = 1.0/(pow(ces.eigenvalues()[ip]-ces.eigenvalues()[iq],2.0)+myI*1e-9);
       _chern += Theta1*Theta2*temp.real();
     }
   }
 }
     _chern = -2.0*_chern/2/M_PI*gauss_w_ky[nky]* gauss_w_kx[nkx];
    }
  }
}
void cChern::update_kxky(double kx, double ky){
	double xi = kx*kx + ky*ky - _mu;
	int p;
	for (int i = 0; i < pblock; ++i) {
		p = i-_PMAX;
		// only lower left part of the matrix is needed for storage.
		_bdg_H(i,i) 	   = complex<double>(xi+_h+2*M_PI*p/_T,0.0);
		_bdg_H(i+pblock,i)   = complex<double>(_v*kx,_v*ky);
		_bdg_H(i+pblock,i+pblock)= complex<double>(xi-_h+2*M_PI*p/_T,0.0);
		_bdg_H(i+2*pblock,i+2*pblock) = complex<double>(-(xi+_h+2*M_PI*p/_T),0.0);
		_bdg_H(i+3*pblock,i+2*pblock) = complex<double>(_v*kx,-_v*ky);
		_bdg_H(i+3*pblock,i+3*pblock) = complex<double>(-(xi-_h+2*M_PI*p/_T),0.0);
	}
}
