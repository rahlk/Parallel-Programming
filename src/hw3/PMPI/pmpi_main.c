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

int *nevents;

// int MPI_Init( int *argc, char ***argv );
//
// int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
//
// int PMPI_Finalize(void);
int MPI_Init( int *argc, char ***argv )
{
  int init;
  init = PMPI_Init(argc, argv);
  printf("in my MPI_Init wrapper\n");
  //... Initializing array
  int rank, nprocs;

  PMPI_Comm_rank (MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  nevents=(int*)malloc(sizeof(int)*nprocs);
  memset(nevents, 0, sizeof(int)*nprocs); // Initialize the array

  //call the actual init function.
  return init;
}

int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {

  *(nevents+dest)+=1;
  return  PMPI_Isend(buf, count, datatype, dest, tag, comm,request);
}

int PMPI_Finalize(void) {
  main();
  return PMPI_Finalize();
}

int main(){
  int rank, numproc;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numproc);

  if (rank==0){
    int i, j, *all;
    MPI_Status *stat;
    all=(int*)malloc(sizeof(int)*numproc*numproc);
    stat = (MPI_Status*)malloc(sizeof(MPI_Status)*(numproc));

    for(i=1; i<numproc; i++) {
      PMPI_Recv(all+i*numproc, numproc, MPI_INT, i, 1, MPI_COMM_WORLD, stat+i);
    }

    for (i=1; i<numproc; i++) {
      printf("Rank %d ", i);
      for (j=0; j<numproc; j++) {
        if(j!=i) {
          int loc = i*numproc+j;
          printf("%d ",all[loc]);
        }
      }
      printf("\n");
    }
  }

  else {
    MPI_Request *req;
    req=(MPI_Request *)malloc(sizeof(MPI_Request)*numproc);
    PMPI_Isend(&nevents, numproc, MPI_CHAR, 0, 1, MPI_COMM_WORLD, req+numproc);
  }
  return(0);
}
