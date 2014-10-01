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
  stride = pblock4*recvcount;
  for(int ig = 0; ig<size; ++ig) {
    if (ig ==rank){
      chern_rank = complex<double> (0.0,0.0);
      cout << "rank" << ig << "has started"<< endl;
      curvature_rank = new double[recvcount];
      bdgE_rank = new double[stride];
      for (int i=0; i<recvcount; ++i) {
	clock_t start = clock(); 
        update(recvbuf[i]);
	clock_t end = clock(); 
	if (rank==root) cout << "task " << recvbuf[i] <<"out of " << recvcount << "used " << double (end-start)/ (double) CLOCKS_PER_SEC  << endl; 
        chern_rank += _chern;
	curvature_rank[i] = _temp_curv;
	for(int ibdgE = 0;ibdgE<pblock4;++ibdgE) {
	  bdgE_rank[i*pblock4+ibdgE] = _bdg_E(ibdgE);
	}
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
    curvature = new double [_NKX2];
    bdgE = new double [_NKX2*pblock4];
    recvcounts = new int[size];
    displs_r = new int[size];
    recvcountsE = new int[size];
    displs_rE = new int[size];
    offset = 0; offsetE = 0;
    for(int ig = 0;ig<size;++ig){
      recvcounts[ig] = compute_count(ig,size);
      displs_r[ig] = offset;
      offset += recvcounts[ig];
      recvcountsE[ig] = compute_count(ig,size)*pblock4;
      displs_rE[ig] = offsetE;
      offsetE += recvcountsE[ig];
    }
  }
  MPI_Gatherv(curvature_rank,recvcount,MPI_DOUBLE,curvature,recvcounts,displs_r,MPI_DOUBLE,root,COMM_WORLD);
  MPI_Gatherv(bdgE_rank,stride,MPI_DOUBLE,bdgE,recvcountsE,displs_rE,MPI_DOUBLE,root,COMM_WORLD);
  if (root==rank) {
    ofstream curv_output,AKX,AKY;
    curv_output.open("curvature.OUT");    assert(curv_output.is_open());
    AKX.open("AKX.OUT");    assert(AKX.is_open());
    AKY.open("AKY.OUT");    assert(AKY.is_open());
    for(int nkx = 0;nkx <_NKX;++nkx) {
      AKX << gauss_k[nkx] << endl;
      AKY << gauss_k[nkx] << endl;
    }
    for(int nky = 0;nky <_NKX;++nky){
      for(int nkx = 0;nkx <_NKX;++nkx){
	curv_output << curvature[nkx+nky*_NKX] << '\t';
      }
      curv_output << endl;
    }
    curv_output.close();AKX.close();AKY.close();
    /*
    ofstream bdgE_output;
    bdgE_output.open("bdgE.OUT");
    assert(bdgE_output.is_open());
	for(int ibdgE=0;ibdgE<pblock4;++ibdgE){
	  bdgE_output << bdgE[nk*pblock4+ibdgE] << '\t';

      }
      bdgE_output << endl;
    }
    bdgE_output.close();
    */
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
  } else {
    int lowerbound = 0;
    int upperbound = 1;
    int nkx = nk % _NKX;     // --> the modulo (because nk = nkx+ nky * NKX )
    int nky = int (nk/_NKX); // --> the floor
    //    double kx = -kmax + nkx * kmax *2.0 /(_NKX-1);
    //    double ky = -kmax + nky * kmax *2.0 /(_NKX-1);
    double kx = gauss_k[nkx];
    double ky = gauss_k[nky];
    SelfAdjointEigenSolver<MatrixXcd> ces;
    complex<double> u,v,up,vp, Theta1,Theta2, temp;
    complex<double> myI (0.0,1.0);
    update_kxky(kx,ky);
    ces.compute(_bdg_H);
    _bdg_E = ces.eigenvalues(); // assuming eigenvalues are sorted in ascending order, but could be wrong since Eigen library does not gurantee that... Oops... Good luck!
    _bdg_V = ces.eigenvectors();
    /*for(int ip = 0; ip < 2*pblock;++ip){
      if (_bdg_E[ip]/(M_PI/_T) >= -1.1) {
	lowerbound = ip;
	break;
      }
    }
    for(int ip = 2*pblock; ip < pblock4;++ip){
      if (_bdg_E[ip]/(M_PI/_T) >= 1.1) {
	upperbound = ip;
	break;
      }
      }*/
    if (lowerbound < 0 || upperbound < 0){
      _temp_curv = 0.0;
      _chern = complex<double> (0.0,0.0); 
      //      cout << "no contribution is added" << endl;
    } else {
      // cout  <<"lower bound = " << lowerbound << " upper bound = " << upperbound << ", and " << upperbound-lowerbound << " is considered for computation." <<endl;
      _chern = complex<double> (0.0,0.0);
      for(int ih = lowerbound; ih < pblock; ++ih) { // hole branch contribution 
	  for(int ip = pblock;ip<=upperbound;++ip){ // particle branch contribution
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
  int p;
  for (int i = 0; i < pblock; ++i) {
    p = i-_PMAX;
    // only lower left part of the matrix is needed for self-adjoint matrix storage.
    _bdg_H(i*2,i*2)     = complex<double>(_mu-_J-2.0*_b*(2.0-cos(kx)-cos(ky))+_J*cos(kx)*cos(ky),0.0);
    _bdg_H(i*2+1,i*2)   = complex<double>(_a*sin(kx),_a*sin(ky));
    _bdg_H(i*2+1,i*2+1) = complex<double>(-_mu+_J+2.0*_b*(2.0-cos(kx)-cos(ky))-_J*cos(kx)*cos(ky),0.0);
  }
}
