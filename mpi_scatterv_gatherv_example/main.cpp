#include <mpi.h>
#include <iostream>
using namespace std;
using namespace MPI;
#define pblock 2
#define _NKX 12
int compute_count(int rank, int size){
  // compute distribution count: the last rank does the remaineder job while the rest do the most even work. 
  int result;
  if (rank != size-1) {
    result = int(_NKX/size);
  } else {
    result = int(_NKX/size) + _NKX%size;
  }
  return result;
}

int main(int argc, char** argv){
  int rank, size, recvcount, sendcount, stride;
  int *sendbuf, *recvbuf;
  int *sendcounts, *displs, *recvcounts, *displs_r;
  double  *localEig, *TotalEig;
  const int root = 0;
  int offset;
  Init(argc, argv);
  rank = COMM_WORLD.Get_rank();
  size = COMM_WORLD.Get_size();
  if (rank == root){ // send process is only root significant
    sendbuf = new int[_NKX];
    for(int i = 0; i< _NKX; ++i){
      sendbuf[i] = i;
    }
    sendcounts = new int[size];
    displs = new int[size];
    for(int i=0; i<size; i++){
      sendcounts[i] = compute_count(i,size);
      displs[i] = i*int(_NKX/size);
    }
  }
  recvcount = compute_count(rank,size); // This is a rank dependent variable. 
  recvbuf = new int[recvcount]; // So is this array: rank dependent size
  MPI_Scatterv(sendbuf,sendcounts,displs,MPI_INT,recvbuf,recvcount,MPI_INT,root,COMM_WORLD);
  stride = pblock*recvcount;
  localEig = new double[stride];
  for(int i = 0; i<recvcount; ++i){
    for (int j = 0; j<pblock; ++j){
      localEig[i*pblock+j] = recvbuf[i]*3.5+j;
    }
  }

  for(int ig =0; ig<size; ++ig) {
    if (ig==rank) {
      cout << "rank = " << rank << '\t' << "localEig = ";
      for(int i = 0; i<recvcount; ++i){
	for (int j = 0; j<pblock; ++j){
	  cout << localEig[i*pblock+j] << " ";
	}
      }
      cout << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD); 
  }

  if (root==rank) {
    TotalEig = new double [_NKX*pblock];
    recvcounts = new int[size];
    displs_r = new int[size];
    offset = 0;
    for(int ig=0;ig<size;++ig){
      recvcounts[ig] = compute_count(ig,size)*pblock;
      displs_r[ig] = offset;
      offset += recvcounts[ig]; 
    }
  }
  MPI_Gatherv(localEig, stride, MPI_DOUBLE, TotalEig, recvcounts, displs_r, MPI_DOUBLE, root, COMM_WORLD);
  if (rank == root) {
    for(int i =0;i<_NKX;++i){
      for (int j = 0; j<pblock;++j){
        cout << TotalEig[i*pblock+j] << " ";
      }
      cout << endl;
    }
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
  return 0;
}
