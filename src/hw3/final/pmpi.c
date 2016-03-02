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

// ...Global variables.
int *nevents, rank, numproc;

int MPI_Init( int *argc, char ***argv )
{
  int init;
  init = PMPI_Init(argc, argv);

  PMPI_Comm_rank (MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numproc);

  //... Initialize array
  nevents=(int*)malloc(sizeof(int)*numproc);
  memset(nevents, 0, sizeof(int)*numproc);

  //... Call the actual init function.
  return init;
}

int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {

  *(nevents+dest)+=1; // Record the number of events
  return  PMPI_Isend(buf, count, datatype, dest, tag, comm,request);
}

int MPI_Finalize(void) {

  if (rank==0){
    int i, j, *all;

    // Setup outfile file handler
    FILE *f = fopen("./matrix.data", "w");
    if (f==NULL) {
      exit(1);
    }

    // Print Rank0's Isend frequencies
    fprintf(f, "%d ", rank);
    for(i=0; i<numproc; i++) {
      fprintf(f, "%d ", nevents[i]);
    }
    fprintf(f, "\n");


    // Set up MPI Receive params.
    MPI_Status *stat;
    all=(int*)malloc(sizeof(int) * numproc); // all is a 1d array that holds...
    //...the received data. Note: all is volatile is is reset for every rank.

    stat = (MPI_Status*)malloc(sizeof(MPI_Status)*(numproc));

    for(i=0; i<numproc-1; i++) {
      memset(all, 0,   sizeof(int) * numproc);
      PMPI_Recv(all, numproc, MPI_INT, i+1, 1, MPI_COMM_WORLD, stat+i);
      fprintf(f, "%d ", i+1);
      for (j=0; j<numproc; j++) {
        fprintf(f, "%d ", all[j]);
      }
      fprintf(f, "\n");
    }
  }

  else {
    MPI_Request *req;
    req=(MPI_Request *)malloc(sizeof(MPI_Request)*numproc);
    // Send info of all my Isends to rank0.
    PMPI_Send(nevents, numproc, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  return PMPI_Finalize();
}
