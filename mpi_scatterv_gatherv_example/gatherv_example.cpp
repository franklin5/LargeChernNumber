#include <mpi.h>
#include <iostream>
using namespace std;
using namespace MPI;

int main(int argc, char** argv){
  
  int gsize,*sendarray; 
  int *rbuf, stride; 
  int *displs,i,*rcounts; 
  const int root = 0;
  int rank;
  Init(argc, argv);
  rank = COMM_WORLD.Get_rank();
  gsize = COMM_WORLD.Get_size();
  for (int ig=0;ig<gsize;++ig) {
    if(rank == ig) {
      sendarray = new int[10+rank];
      for(int i=0;i<10+rank;++i) {
	sendarray[i] = i+ig;
	//cout << sendarray[i] << " ";
      }
      //      cout << endl;
    }
  }
  /*
  for (int ig=0;ig<gsize;++ig) {
    if(rank == ig) {
      for(int i=0;i<10+rank;++i){
	cout << sendarray[i] << " ";
      }
      cout << endl;
      MPI_Barrier(COMM_WORLD);
    }
  }
  */
  stride = rank+10;
  if(rank == root) {
    rbuf = new int[gsize*(gsize+19)/2];
  }
  displs = new int[gsize];
  rcounts = new int[gsize];
  if (rank==root){
    int offset = 0;
  for (int i=0; i<gsize; ++i) { 
    displs[i] = offset;
    rcounts[i] = i+10;
    offset += rcounts[i]; 
  } 
  }
  MPI_Gatherv( sendarray, stride, MPI_INT, rbuf, rcounts, displs, MPI_INT, root, COMM_WORLD);
  if (rank == root) {
      int offset = 0;
    for(int i =0;i<gsize;++i){
      for (int j = 0; j<i+10;++j){
	cout << rbuf[offset+j] << " ";
      }
      offset += i+10;
      cout << endl;
      }
  }
  if (rank == root) {
  delete []rbuf;
  delete []displs;
  delete []rcounts;
  }
  delete []sendarray;

  Finalize();
  return 0;
}
