/*
Author: Rahul Krishna
unity: rkrish11
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

//
// int MPI_Isend(void *msg, count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* req, int* ncount);

int main(int argc, char *argv[]) {

  int   numproc, rank, len;
  char  hostname[MPI_MAX_PROCESSOR_NAME];

  PMPI_Init(&argc, &argv);
  PMPI_Comm_size(MPI_COMM_WORLD, &numproc);
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Get_processor_name(hostname, &len);

  if (rank==0) {
    int *freq,i,j;
    freq=(int *)malloc(sizeof(int)*numproc);
    char *temp;
    temp=(char*)malloc(sizeof(char)*(numproc-1));
    MPI_Status *stat, *stat1;
    stat = (MPI_Status*)malloc(sizeof(MPI_Status)*(numproc-1));
    stat1 = (MPI_Status*)malloc(sizeof(MPI_Status)*(numproc-1));
    MPI_Request *req;
    req = (MPI_Request *)malloc(sizeof(MPI_Request)*(numproc-1));
    int N=numproc*numproc;

    for(i=1; i<numproc; i++) {
        PMPI_Recv(temp+i-1, 1, MPI_CHAR, i, 0, MPI_COMM_WORLD, stat+(i-1));//, req+(i-1)*2);
    }

    for(i=1; i<numproc; i++) {
      PMPI_Recv(freq+i*numproc, numproc, MPI_INT, i, 1, MPI_COMM_WORLD,
       stat1+(i-1));
    }

    printf("echo\n");
    // MPI_Waitall((numproc-1), req, stat);
    for (i=1; i<numproc; i++) {
      printf("Rank %d ", i);
      for (j=0; j<numproc; j++) {
        if(j!=i) {
          int loc = i*numproc+j;
          printf("%d ",freq[loc]);
        }
      }
      printf("\n");
    }
  }

  else {
    int i, *nsend;
    char *rMsg, msg='x';
    rMsg=(char*)malloc(sizeof(char));
    nsend=(int*)malloc(sizeof(int)*numproc);
    // msg=(char*)malloc(sizeof(char));
    // memset(msg, 'z', sizeof(char));
    memset(nsend, 0, sizeof(int)*numproc);
    MPI_Request *req;
    req = (MPI_Request *)malloc(sizeof(MPI_Request)*(numproc));
    MPI_Status *stat;
    stat = (MPI_Status*)malloc(sizeof(MPI_Status)*(numproc-1));
    for (i=0; i<numproc; i++) {
      if(i!=rank) {
        *(nsend+i)+=*(nsend+i)+1;
        PMPI_Isend(&msg, 1, MPI_CHAR, i, 0, MPI_COMM_WORLD, &(req[i]));
      }
    }
    // printf("Echo-1\n");
    for (i=1; i<numproc; i++) {
      if (i!=rank)
        PMPI_Recv(rMsg, 1, MPI_CHAR, i, 0, MPI_COMM_WORLD, stat+i-1);
    }
    // printf("Echo-2\n");
    MPI_Isend(nsend, numproc, MPI_INT, 0, 1, MPI_COMM_WORLD, req+numproc);
    // MPI_Isend(msg, 1, MPI_CHAR, i, 0, MPI_COMM_WORLD, req+numproc);
    // printf("Echo-3\n");
  }
  PMPI_Finalize();
  return(0);
}

// int MPI_Isend(void *msg, count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* req, int* ncount) {
//   *ncount=*ncount+1;
//   return PMPI_Isend(msg, count, datatype, dest, tag, comm, req);
// }
