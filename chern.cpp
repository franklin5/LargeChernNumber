/*
 * edge.cpp
 *
 *  Created on: Sep 2, 2014
 *      Author: Lin Dong
 */
#include "chern.h"

void cChern::distribution(){
	int rank, size;
	const int root = 0;
	int *sbuffer = NULL;
	Init(_argc, _argv);
	rank = COMM_WORLD.Get_rank();
	size = COMM_WORLD.Get_size();
	double* TotalEig = NULL;
	if (rank == root){
		sbuffer = new int[_NKX*_NKX];
		for(int i = 0; i< _NKX*_NKX; ++i){
		  sbuffer[i] = i;
		}
	}
	int workload_remainder = (_NKX*_NKX) % (size);
	int workload_even = int ((_NKX*_NKX)/(size)); // in the special case of workload_remainder==0, the last processor is wasted. But should be fine.
	int** recvB = new int*[size];
	double** localEig = new double*[size];
	int *sendcounts = NULL,  *displs = NULL; // Note: sbuffer, sendcounts, displs, stype are significant for the root process only. But we are spreading over all ranks.
	sendcounts = new int[size];  displs = new int[size];
	int *recvcounts = NULL,  *displs_r = NULL; // Note: rbuf, rcounts, displs, rtype are significant for the root process only.
		recvcounts = new int[size];  displs_r = new int[size];
	for (int ig = 0; ig < size; ++ig) {
		displs[ig] = ig*workload_even; // displacement relative to sbuffer
		displs_r[ig] = ig*workload_even*4*pblock; // displacement relative to sbuffer
		if (ig == size-1) {
		  recvB[ig] = new int[workload_remainder];
		  localEig[ig] = new double[workload_remainder*4*pblock];
		  sendcounts[ig] = workload_remainder; // the last processor does the variable amount of work.
		  recvcounts[ig] = workload_remainder*4*pblock;
		} else {
		  recvB[ig] = new int[workload_even];
		  localEig[ig] = new double[workload_even*4*pblock];
		  sendcounts[ig] = workload_even;
		  recvcounts[ig] = workload_even*4*pblock;
		}
	}
	int epp_var;
	if (rank == size-1) {
			epp_var = workload_remainder;
	} else {
			epp_var = workload_even;
	}
	//cout << rank << '\t' << epp_var << endl;
	// This is a good way of chekcing and simulating access to the communicated value.
	//cout << rank << '\t' << sendcounts[rank] << '\t' << displs[rank] << endl;
	//	if (rank == root) {
	//		for (int ig = 0; ig < size; ++ig) {
	//			for (int temp = displs[ig]; temp < displs[ig]+sendcounts[ig]; ++temp) {
	//				cout << sbuffer[temp] << '\t';
	//			}
	//			cout << endl;
	//		}
	//
	//	}
	// unevenly distributed:
	 MPI_Scatterv(sbuffer, sendcounts, displs, MPI_INT, &recvB[rank][0], epp_var, MPI_INT, root, COMM_WORLD); // worked without derived datatpe
	 //	displs 	array specifying the displacement relative to sbuffer at which to place the incoming data from corresponding process,
	 // MPI_Scatterv(sbuffer, sendcounts, displs, MPI_INT, rptr, 1, recvtype, root, COMM_WORLD);
	 // TODO: derived datatype implementation?
//		MPI_Datatype recvtype;
//		MPI_Datatype sendtype;
//		MPI_Type_vector( epp_var, 1, epp_var, MPI_INT, 	&recvtype);
//		MPI_Type_vector( epp_var, 1, epp_var, MPI_DOUBLE, &sendtype);
//		MPI_Type_commit( &recvtype );
//		MPI_Type_commit( &sendtype );
	 // helpful examples:
	 // https://www.cac.cornell.edu/ranger/MPIcc/gathervscatterv.aspx
	 // http://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node72.html
	 // http://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node72.html
	 // http://static.msi.umn.edu/tutorial/scicomp/general/MPI/deriveddata/vector_c.html
// This is to check if recvB has received correct value from sbuffer, via scatterv.
//	cout << "rank=" << rank << "recvB=";
//	for (int i=0; i<epp_var; ++i) {
//			 cout << recvB[rank][i] << '\t';
//	}
//	cout << endl;
	SelfAdjointEigenSolver<MatrixXcd> ces;
	for (int i=0; i<epp_var; ++i) {
		//cout << "recvB[" << rank << "][" << i << "] = " << recvB[rank][i] << endl;
		update(recvB[rank][i]);
		//clock_t start = clock();
		ces.compute(_bdg_H,0); // eigenvectors are also computed.
		//ces.compute(_bdg_H); // eigenvectors are also computed.
		//clock_t end = clock();
		//cout << double (end-start)/ (double) CLOCKS_PER_SEC  << endl;
		for(int j = 0; j < 4*pblock; ++j){
		  localEig[rank][j+i*4*pblock]=ces.eigenvalues()[j];
		}
		//cout << "rank " << rank << " is finished out of epp "<< epp_var << endl;
	}
	if(rank == root){
		TotalEig = new double[_NKX*_NKX*4*pblock];
	}
	cout << "dsafasdfwef" << epp_var << endl;
	MPI_Gatherv(&localEig[rank][0], epp_var*4*pblock, MPI_DOUBLE, TotalEig, recvcounts, displs_r, MPI_DOUBLE, root, COMM_WORLD);
	// TODO: gatherv still faces problem. missing values from scatterv.
	// MPI_Gatherv(&localEig[rank][0], 1, sendtype, TotalEig, sendcounts, displs, MPI_DOUBLE, root, COMM_WORLD);
	if (rank == root){
	ofstream bdg_output;
	bdg_output.open("spectrum_2109.OUT"); // TODO: modify output file name
	assert(bdg_output.is_open());
	for(int i = 0; i<size; ++i){
		for(int j = 0; j<epp_var; ++j){
			for(int q = 0; q<4*pblock; ++q){
				bdg_output << TotalEig[i*4*pblock*epp_var+j*4*pblock+q] << '\t';
			}
			bdg_output << endl;
		}
		bdg_output << endl;
	}
	bdg_output.close();
	}
	Finalize();
	for (int ig = 0; ig < size; ++ig) {
		delete []localEig[ig];
		delete []recvB[ig];
	}
	delete []localEig;
	delete []recvB;
	delete []TotalEig;
	delete []sbuffer;
	delete []sendcounts;
	delete []displs;
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
	  while (fscanf(sf_inputR, "%lf", &reD) != EOF && fscanf(sf_inputI, "%lf", &imD) != EOF ){
		Delta_t(count) = complex<double>(reD,imD);
		//cout << Delta_t(count) << '\t' << count << endl;
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
					Gamma2 +=  conj(Delta_t(ig)) *
							complex<double> (cos(2*M_PI*(q-p)*t/_T),-sin(2*M_PI*(q-p)*t/_T));
					t += dt;
				}
	  			Gamma2 = Gamma2/_T*dt;
	  			//cout << Gamma2 << endl;
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
