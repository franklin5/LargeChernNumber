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
      cout << "rank" << ig << "has started"<< endl;
      bdgE_rank = new double[stride];
      for (int i=0; i<recvcount; ++i) {
	clock_t start = clock(); 
        update(recvbuf[i]);
	clock_t end = clock(); 
	if (rank==root) cout << "task " << recvbuf[i] <<"out of " << recvcount << "used " << double (end-start)/ (double) CLOCKS_PER_SEC  << endl; 
	for(int ibdgE = 0;ibdgE<pblock4;++ibdgE) {
	  for(int in = 0;in<_NMAX;++in){
	    bdgE_rank[i*pblock4*_NMAX+ibdgE*_NMAX+in] = _bdg_E(ibdgE*_NMAX+in);
	  }
	}
      }
      cout << "rank " << rank << " has finished "<< recvcount << " tasks, " << endl;
    }
  }
  if (root==rank) {
    bdgE = new double [_NKX2*pblock4*_NMAX];
    recvcounts = new int[size];
    displs_r = new int[size];
    recvcountsE = new int[size];
    displs_rE = new int[size];
    offset = 0; offsetE = 0;
    for(int ig = 0;ig<size;++ig){
      recvcounts[ig] = compute_count(ig,size);
      displs_r[ig] = offset;
      offset += recvcounts[ig];
      recvcountsE[ig] = compute_count(ig,size)*pblock4*_NMAX;
      displs_rE[ig] = offsetE;
      offsetE += recvcountsE[ig];
    }
  }
  MPI_Gatherv(bdgE_rank,stride,MPI_DOUBLE,bdgE,recvcountsE,displs_rE,MPI_DOUBLE,root,COMM_WORLD);
  if (root==rank) {
    ofstream KX;
    KX.open("KX.OUT");    
    ofstream bdgE_output;
    bdgE_output.open("edgeE.OUT");
    assert(bdgE_output.is_open());
    for(int nkx = 0;nkx <_NKX;++nkx) {
      KX << gauss_k[nkx] << endl;
      for(int ibdgE=0;ibdgE<pblock4;++ibdgE){
	for(int in = 0; in<_NMAX; ++in){
	  bdgE_output << bdgE[nkx*pblock4*_NMAX+ibdgE*_NMAX+in] << '\t';
	}
      }
      bdgE_output << endl;
    }
    KX.close();    
    bdgE_output.close();
    /*    complex<double> temp ;
    ofstream bdg;
    bdg.open("H.OUT");
    assert(bdg.is_open());
    for (int ip = 0;ip<pblock4;++ip){
      for (int iq = 0;iq<pblock4;++iq){
	temp = 	_bdg_H(ip,iq)-conj(_bdg_H(iq,ip)) ;
	bdg << temp.imag() << '\t';
      }
      bdg << endl;
    }
    bdg.close();*/
    delete []curvature;
    delete []sendbuf;
    delete []sendcounts;
    delete []displs;
    delete []recvcounts;
    delete []displs_r;
    delete []recvcountsE;
    delete []displs_rE;
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
    //int p, q;
    // The off-diagonal coupling introduced from time-dependent order parameter should be computed only here.
    //    complex<double> Gamma1, Gamma2;
    /*FILE *sf_inputR, *sf_inputI;
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
    fclose (sf_inputI);*/
    //for (int i = 0; i < pblock; ++i) {
      //p = i-_PMAX;
      //for (int j = 0; j <=i; ++j) { 
	//q = j-_PMAX;
	/*Gamma1 = complex<double> (0.0,0.0);
	Gamma2 = complex<double> (0.0,0.0);
	t = 0.0;
	for (int ig = 0; ig < count; ++ig) {
	  Gamma1 += abs(Delta_t(ig)) * complex<double> (cos(2*M_PI*(q-p)*t/_T), sin(2*M_PI*(q-p)*t/_T));
	  Gamma2 += abs(Delta_t(ig)) * complex<double> (cos(2*M_PI*(q-p)*t/_T), sin(2*M_PI*(q-p)*t/_T));
	  t += dt;
	  }*/
	//Gamma1 = complex<double> (1.0,0.0);
	//Gamma2 = complex<double> (1.0,0.0);
	//_bdg_H(i*4,  j*4+3) = -Gamma1;
	//_bdg_H(i*4+1,j*4+2) =  Gamma1;
	//_bdg_H(i*4+2,j*4+1) =  Gamma2;
	//_bdg_H(i*4+3,j*4)   = -Gamma2;
    //}
    //}
    double Lambda;
    int n,m;
    for (int i = 0; i < pblock; ++i) {
      for(int in = 0; in < _NMAX;++in){
	n = 1+in;
	for(int im = 0; im < in; ++im){
	  m = 1+im;
	  if (in!=im){
	    Lambda = 2.0*_a/_L*m*n*(1-pow(-1.0,m+n))/(n*n-m*m);
	    _bdg_H(i*pblock4*_NMAX+in*2,i*pblock4*_NMAX+im*2+1) = -Lambda; 
	    _bdg_H(i*pblock4*_NMAX+in*2+1,i*pblock4*_NMAX+im*2) = Lambda;
	  }
	}
      }
    } 
  } else {
    int nkx = nk;
    double kx = gauss_k[nkx];
    SelfAdjointEigenSolver<MatrixXd> ces;
    update_kxky(kx);
    ces.compute(_bdg_H,0);
    _bdg_E = ces.eigenvalues();
  }
}


void cChern::update_kxky(double kx){
  int p,n;
  for (int i = 0; i < pblock; ++i) {
    p = i-_PMAX;
    for(int in = 0; in < _NMAX;++in){
      n = in + 1;
      // only lower left part of the matrix is needed for self-adjoint matrix storage.
      _bdg_H(i*2*_NMAX+in*2,i*2*_NMAX+in*2)     = _mu-_J-2.0*_b*(0.5*kx*kx+0.5*pow(n*M_PI/_L,2.0))+_J*(1-0.5*kx*kx)*(1-0.5*pow(n*M_PI/_L,2.0));
      _bdg_H(i*2*_NMAX+in*2+1,i*2*_NMAX+in*2)   = _a*kx;
      _bdg_H(i*2*_NMAX+in*2+1,i*2*_NMAX+in*2+1) = -_mu+_J+2.0*_b*(0.5*kx*kx+0.5*pow(n*M_PI/_L,2.0))-_J*(1-0.5*kx*kx)*(1-0.5*pow(n*M_PI/_L,2.0));
    }
  }
}
