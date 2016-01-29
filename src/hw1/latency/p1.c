#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h> /* memset */

#define	NUMBER_REPS	1000

float std(double data[], int n);

int main (int argc, char *argv[])
{

// MPI variables
int reps, tag, numtasks, rank, dest, source, rc, n, i=3;
float avgT, stdev;

// Stats stuff...
double Tstart, Tend, delT, sumT;
MPI_Status status;

// Initialize MPI
MPI_Init(&argc,&argv);

// Assign MPI variables
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);

// Block the caller until all processes in the communicator have called it
MPI_Barrier(MPI_COMM_WORLD);

// time = 0;
tag = 1;
reps = NUMBER_REPS; // Do 1000 repeats - For stats.

/* Note: Rank 0 sends data, Rank 1 receives it.*/
for (i=5; i<12; i++) {
  int n_char = pow(2,i);
  // char msg = 'x';//[2^i];
  char msg[n_char];
  memset(msg, 'x', n_char*sizeof(char));
  int chunksize = sizeof(msg);
  if (rank == 0) {
    printf("%d \t ", sizeof(char)*sizeof(msg));
    for (dest=1;dest<numtasks;dest++) {
      sumT=0;
      double tarray[NUMBER_REPS];
      for (n = 0; n < reps; n++) {
        // Initialize MPI clock
        Tstart = MPI_Wtime();
        rc = MPI_Send(&msg, chunksize, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        rc = MPI_Recv(&msg, chunksize, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &status);
        Tend = MPI_Wtime();
        delT = Tend - Tstart;
        sumT+=delT;
        tarray[n]=delT;
        }
        avgT = (sumT)/reps;
        stdev = std(tarray, reps);
        printf("%0.2e %0.2e ",avgT, stdev);
      }
    printf("\n");
    }
  else {
     dest = 0;
     source = 0;
     for (n = 1; n <= reps; n++) {
        rc = MPI_Recv(&msg, chunksize, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
        rc = MPI_Send(&msg, chunksize, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        }
     }
}

MPI_Finalize();
exit(0);
}

float std(double data[], int n) {
    float mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;i++)
      sum_deviation+=(data[i]-mean);
    return sqrt(sum_deviation/(n-1));
}
