#include <mpi.h>
#include <iostream>

using namespace std;
using namespace MPI;
#define _NKX 21

int main(int argc, char** argv){
  
  Init(argc, argv);
  
  int rank, size, recvcount, *sendbuf, *recvbuf, *sendcounts, *displs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int root = 0;

  /* send buffer */
  if (rank==root){
    sendbuf = new int[_NKX];
    for(int i=0; i<_NKX; i++)
      sendbuf[i] = i;
  }

  /* recvcount of each process & sendcounts & displs */

  sendcounts = new int[size];
  displs = new int[size];

  if (rank != size-1) {
    recvcount = int(_NKX/size);
  } else {
    recvcount = int(_NKX/size) + _NKX%size;
  }
  for(int i=0; i<size; i++){
    if (i == size-1) {
      sendcounts[i] = int(_NKX/size) + _NKX%size;
    } else {
      sendcounts[i] =  int(_NKX/size);
    }
    displs[i] = i*int(_NKX/size);
  }

  /* receive buffer */
  recvbuf = new int[recvcount];
  /* Derived datatype approach. Can't make it work...  
  MPI_Datatype recvtype;
  MPI_Type_vector(recvcount,1,recvcount,MPI_INT,&recvtype);
  MPI_Type_commit(&recvtype);  
  MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT, recvbuf, 1, recvtype, root, MPI_COMM_WORLD);
  */
  MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT, recvbuf, recvcount, MPI_INT, root, MPI_COMM_WORLD);

  /* print out */
  for(int i=0; i<size; i++){
    if (rank==i){
      cout << "rank=" << rank << ": " << "recvbuf = [ ";
      for (int i=0; i<recvcount; ++i) {
	cout << recvbuf[i] << " ";
      }
      cout << "]" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  Finalize();
  return 0;
}
