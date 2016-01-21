/******************************************************************************
* FILE: mpi_latency.c
* DESCRIPTION:
*   MPI Latency Timing Program - C Version
*   In this example code, a MPI communication timing test is performed.
*   MPI task 0 will send "reps" number of 1 byte messages to MPI task 1,
*   waiting for a reply between each rep. Before and after timings are made
*   for each rep and an average calculated when completed.
* AUTHOR: Blaise Barney
* LAST REVISED: 04/13/05
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#define	NUMBER_REPS	1000

float std(float data[]);

int main (int argc, char *argv[])
{

// MPI variables
int reps, tag, mpitasks, rank, dest, source, rc, n;
float avgT;

// Stats stuff...
double t, delT, sumT;
float tarray[NUMBER_REPS];
// Data 32bits...2M
char msg32b[4], msg64b[8], msg128b[16], msg256b[32]
msg512b[64], msg1M[128], msg2M[256];

MPI_Status status;

// Initialize MPI
MPI_Init(&argc,&argv);

// Assign MPI variables
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);

// Block the caller until all processes in the communicator have called it
MPI_Barrier(MPI_COMM_WORLD);

time = 0;
tag = 1;
reps = NUMBER_REPS; // Do 1000 repeats - For stats.

/* Note: Rank 0 sends data, Rank 1 receives it.*/

if (rank == 0) {
  dest = 1;
  source = 1;
  printf("Size, Mean, Stdev\n");
  for (i=2; i<9; i++) {
    int n_char = 2^i;
    char msg[n_char] = {'x'};
    int chunksize = sizeof(msg);
    for (n = 1; n <= reps; n++) {
      // Initialize MPI clock
      t = MPI_Wtime();
      rc = MPI_Send(&msg, chunksize, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
      rc = MPI_Recv(&msg, chunksize, MPI_CHAR, source, tag, MPI_COMM_WORLD,
                    &status);
      delT = MPI_Wtime() - t;
      sumT += delT;
      tarray[n]=(float) delT
      }
     avgT = (sumT*1000000)/reps;
     stdev = std(tarray);
     printf("%d, %0.2f, %0.2f",n_char, avgT, stdev);
    }
  }


else if (rank == 1) {
   dest = 0;
   source = 0;
   for (n = 1; n <= reps; n++) {
      rc = MPI_Recv(&msg, 1, MPI_BYTE, source, tag, MPI_COMM_WORLD, &status);
      rc = MPI_Send(&msg, 1, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
      }
   }

MPI_Finalize();
exit(0);
}

float std(float data[]) {
    float mean=0.0, sum_deviation=0.0;
    int i, n = sizeof(data);
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/n);
}
