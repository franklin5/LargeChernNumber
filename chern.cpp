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
  const int root = 0;
  int offset;
  SelfAdjointEigenSolver<MatrixXcd> ces;
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
      for (int i=0; i<recvcount; ++i) {
	//cout << "rank = " << ig << "recvbuf[" << i << "] = " << recvbuf[i] << endl;
	update(recvbuf[i]);
	//clock_t start = clock();
	ces.compute(_bdg_H,0); // eigenvectors are also computed.
	//	ces.compute(_bdg_H); // eigenvectors are also computed.
	//cout << _bdg_H << endl;
	//clock_t end = clock();
	//cout << double (end-start)/ (double) CLOCKS_PER_SEC  << endl;
	for(int j = 0; j < pblock4; ++j){
	  localEig[j+i*pblock4]=ces.eigenvalues()[j];
	  //	  cout << localEig[j+i*pblock4] << " ";
	}
	//	cout << endl;
	//	cout << ces.eigenvalues() << endl;
      }
          cout << "rank " << rank << " has finished "<< recvcount << "tasks." << endl;
    }
    //    MPI_Barrier(COMM_WORLD);
  }
  if (root==rank) {
    TotalEig = new double [_NKX2*pblock4];
    recvcounts = new int[size];
    displs_r = new int[size];
    offset = 0;
    for(int ig=0;ig<size;++ig){
      recvcounts[ig] = compute_count(ig,size)*pblock4;
      displs_r[ig] = offset;
      offset += recvcounts[ig];
      //cout << offset << " ";
    }
    //    cout << endl;
  }
  MPI_Gatherv(localEig, stride, MPI_DOUBLE, TotalEig, recvcounts, displs_r, MPI_DOUBLE, root, COMM_WORLD);
  if (rank == root) {
    int itemp;
    ofstream bdg_output;
    bdg_output.open("spectrum_2109.OUT"); // TODO: modify output file name
    assert(bdg_output.is_open());
    for(int i =0;i<size;++i){    
      itemp = compute_count(i,size);
      if( i != size-1) {
	offset = itemp;
      } else {
	offset = offset; // offset is updated as size-2 position
      }
      //      cout << itemp << endl;
      for (int j = 0; j<itemp;++j){
	for (int q = 0; q<pblock4;++q){
//	  cout << TotalEig[i*offset*pblock4+j*pblock4+q] << '\t';
	  //	  cout << i*offset*pblock4+j*pblock4+q << '\t';
	  bdg_output << TotalEig[i*offset*pblock4+j*pblock4+q] << '\t';
	}
//	cout << endl;
	bdg_output << endl;
      }
      //      bdg_output << endl;
    }
    bdg_output.close();
  }
  if (root==rank) {
    delete []sendbuf;
    delete []sendcounts;
    delete []displs;
    delete []TotalEig;
    delete []recvcounts;
    delete []displs_r;
  }
  delete []recvbuf;
  delete []localEig;
  Finalize();
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
	  sf_inputR = fopen ("Rdata_2109.dat","r"); // TODO: modify input file name
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
	  int nkx = nk % _NKX;     // --> the modulo (because nk = nkx+ nky * NKX )
	  int nky = int (nk/_NKX); // --> the floor
	  double kmax = 2.0; // TODO: modify momentum space cutoff value
	  double kx = -kmax + nkx * kmax *2.0 /(_NKX-1);
	  double ky = -kmax + nky * kmax *2.0 /(_NKX-1);
	  //	  cout << "kx = " << kx << ", " << "ky = " << ky << endl;
	  // This is to test the bulk spectrum periodicity.
	  update_kxky(kx,ky);

	  /* This is left for the Chern number calculation using Ming's method.
	  double dk = 0.5 * kmax * 2.0 /(_NKX-1);
	  double q1x, q2x, q1y, q2y; // This is to form a small square loop. Idea taken from Ming Gong's code.
	  for (int i = 0; i < 4; ++i) {
		switch (i) {
			case 1:
				q1x = kx - dk;
				q1y = ky - dk;
				q2x = kx + dk;
				q2y = ky - dk;
				break;
			case 2:
				q1x = kx + dk;
				q1y = ky - dk;
				q2x = kx + dk;
				q2y = ky + dk;
				break;
			case 3:
				q1x = kx + dk;
				q1y = ky + dk;
				q2x = kx - dk;
				q2y = ky + dk;
				break;
			case 4:
				q1x = kx - dk;
				q1y = ky + dk;
				q2x = kx - dk;
				q2y = ky - dk;
				break;
			default:
				break;
		}
	  }
	  update_kxky(q1x,q1y);

	  update_kxky(q2x,q2y);
	  */
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
