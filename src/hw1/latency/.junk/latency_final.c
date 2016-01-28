#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h> /* memset */

#define	NUMBER_REPS	1

float std(double data[], int n);

int main (int argc, char *argv[])
{

// MPI variables
int reps, tag, numtasks, rank, rank0, rank1, to, dest, source, rc, n, i, j;
float avgT, stdev;

// Stats stuff...
double Tstart, Tend, delT, sumT, stdT;
printf("Data_Size Mean Stdev\n");

MPI_Status status;
MPI_Request rq;

// Initialize MPI
MPI_Init(&argc,&argv);

// Assign MPI variables
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);

// Block the caller until all processes in the communicator have called it
MPI_Barrier(MPI_COMM_WORLD);
// double *tdarr;
// time = 0;
tag = 1;
reps = NUMBER_REPS; // Do 1000 repeats - For stats.
double Stdarr[numtasks];
double tdarr[numtasks];
double udarr[numtasks];
// tdarr[rank]=0;
// printf("Rank %d\t",rank);
for (i=2;i<11;i++) {
  int n_char = pow(2,i);
  char msg[n_char];
  memset(msg, 'x', n_char*sizeof(char));
  int chunksize=sizeof(msg);
  if (rank==0) {
    double statArr0[9][numtasks-1];
    double statArr1[9][numtasks-1];

    for (rank1=1; rank1<numtasks; rank1++) {
      // double tarray[numtasks];
      sumT=0;
      int rep;
      // for (rep=0; rep<NUMBER_REPS; rep++) {
        Tstart = MPI_Wtime();
        rc = MPI_Send(&msg, chunksize, MPI_CHAR, rank1, tag, MPI_COMM_WORLD);
        rc = MPI_Recv(&msg, chunksize, MPI_CHAR, rank1, tag, MPI_COMM_WORLD, &status);
        Tend=MPI_Wtime();
        delT = Tend-Tstart;
        Stdarr[rep]=delT;
        sumT+=delT;
      // }
    statArr0[i-2][rank1-1]=std(Stdarr, reps);
    statArr1[i-2][rank1-1]=sumT/reps;
    //
    // printf("%d\t%0.2e %0.2e\t", i*sizeof(msg), udarr[rank1-1], tdarr[rank1-1]);
    }
  }
  else {
    dest=rank0;
    source=rank0;
    int chunksize = sizeof(msg);
    rc = MPI_Send(&msg, chunksize, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
    rc = MPI_Recv(&msg, chunksize, MPI_CHAR, 0, rank, MPI_COMM_WORLD, &status);
    }
  }
printf("\n");

  // stdT=std(tdarr, numtasks);
  // printf("%d, %d\n", sizeof(tdarr), sizeof(udarr));
  // for (i=0; i<numtasks; i++)
  //   if (i!=rank)
  //       printf("%e %e  ", udarr[i], tdarr[i]);
  // printf("\n");
  // printf("%f",sumT/numtasks);
  // for (i=0;i<numtasks;i++)
  //   if (i!=rank)
  //     printf("%d:%0.2e ",i,tarray[i]);
// printf("\n");
// printf("%e, %e\n", sumT/numtasks, stdT);
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
    mean=mean/(n-1);
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/n);
}
