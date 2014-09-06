#include <mpi.h>
#include <iostream>
using namespace std;
using namespace MPI;
#define _NKX 11
int compute_count(int rank, int size){
  int result;
  if (rank != size-1) {
    result = int(_NKX/size);
  } else {
    result = int(_NKX/size) + _NKX%size;
  }
  return result;
}

int main(int argc, char** argv){
  int rank, size, recvcount, sendcount, *sendbuf, *recvbuf, *sendcounts, *displs, *recvcounts, *displs_r;
  double *TotalEig;
  double *localEig;
  const int root = 0;
  Init(argc, argv);
  rank = COMM_WORLD.Get_rank();
  size = COMM_WORLD.Get_size();


  if (rank == root){
    sendbuf = new int[_NKX];
    for(int i = 0; i< _NKX; ++i){
      sendbuf[i] = i;
    }
  }
  sendcounts = new int[size];
  displs = new int[size];

  recvcount = compute_count(rank,size);
  /*
  if (rank != size-1) {
    recvcount = int(_NKX/size);
  } else {
    recvcount = int(_NKX/size) + _NKX%size;
  }
  */
  for(int i=0; i<size; i++){
    sendcounts[i] = compute_count(i,size);
    displs[i] = i*int(_NKX/size);
  }
  /*
  for(int i=0; i<size; i++){
    if (i == size-1) {
      sendcounts[i] = int(_NKX/size) + _NKX%size;
    } else {
      sendcounts[i] =  int(_NKX/size);
    }
    displs[i] = i*int(_NKX/size);
  }
  */
  recvbuf = new int[recvcount];

  MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT, recvbuf, recvcount, MPI_INT, root, COMM_WORLD); 
  
  for(int i =0; i<size; i++) {
    if (i==rank) {
      cout << "rank = " << rank << '\t' << "recvbuf = ";
      for ( int j = 0; j < recvcount; ++j){
	cout << recvbuf[j] << '\t';
      }
      cout << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD); // This is necessary for the cout stream to finish one at a time: One use of MPI_Barrier is for example to control access to an external resource such as the filesystem, which is not accessed using MPI.That way, you can be sure that no two processes are concurrently calling cout stream. Remember: One at a time!!!
  }

  int pblock =2;
  //  localEig = new double[recvcount*pblock];
  for(int ig =0; ig<size; ++ig) {
    if (ig==rank) {  
      localEig = new double[recvcount*pblock];
      for(int i = 0; i<recvcount; ++i){
	for (int j = 0; j<pblock; ++j){
	  localEig[i*recvcount+j]=3.5*(10+j)*i;
	  cout << localEig[i*recvcount+j] << '\t' ;
	  //	  cout << i*recvcount+j << '\t';
	}
      }
      cout << endl;
      }
    MPI_Barrier(MPI_COMM_WORLD); // For the same reason here. Writing needs to be done one at a time!!! If we don't have cout, then this should be fine. 
  }  

  //  if (rank == root){
    TotalEig = new double [_NKX*pblock];
    //}
  recvcounts = new int[size];
  displs_r = new int[size];

   for (int ig = 0; ig < size; ++ig) {
     recvcounts[ig] = compute_count(ig,size)*pblock;
     displs_r[ig] =  ig*int(_NKX/size)*pblock;
   }
   sendcount = recvcount * pblock;
  //  recvcounts[rank] = sendcounts[rank] * pblock;
  //  displs_r[rank] = displs[rank] * pblock;

  for (int ig = 0; ig < size; ++ig) {
    if (ig==rank) {
      // recvcounts[ig] = sendcounts[ig] * pblock;
      //displs_r[ig] = displs[ig] * pblock;
      cout << recvcounts[ig] << " " << displs_r[ig] << endl;
      //cout << sendcounts[ig] << " " << displs[ig] << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Gatherv(localEig, sendcount, MPI_DOUBLE, TotalEig, recvcounts, displs_r, MPI_DOUBLE, root, COMM_WORLD);
   
  for(int i = 0;i<size;++i){
    if (i == rank) {
      for(int j = 0; j<recvcounts[i];++j){
	cout << TotalEig[displs_r[i]+j] << " " ;
      }
      cout << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  /*

  // Memory management needs extra care! Need to free pointers at corresponding rank. If not done properly, good luck with seg fault.
  if (rank == root){
    
    delete []sendbuf;
  }
  delete []TotalEig;
  delete []recvbuf; 
  delete []localEig;
  delete []sendcounts;
  delete []displs;
  delete []recvcounts;
  delete []displs_r;
  MPI_Barrier(MPI_COMM_WORLD);
  */
  Finalize();
  return 0;
}
