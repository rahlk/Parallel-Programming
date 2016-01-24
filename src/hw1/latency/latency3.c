vim #include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h> /* memset */

#define	NUMBER_REPS	10

float std(float data[], int n);

int main (int argc, char *argv[])
{

// MPI variables
int reps, tag, numtasks, rank, rank0, rank1, to, dest, source, rc, n, i, j;
float avgT, stdev;

// Stats stuff...
double Tstart, Tend, delT, sumT;
// // Data 32bits...2M
// char msg32b[4], msg64b[8], msg128b[16], msg256b[32]
// msg512b[64], msg1M[128], msg2M[256];
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

// time = 0;
tag = 1;
reps = NUMBER_REPS; // Do 1000 repeats - For stats.
float tarray[numtasks][numtasks];
/* Note: Rank 0 sends data, Rank 1 receives it.*/
for (rank0 = 0; rank0 < numtasks; rank0++) {
    char msg = 'x';
    if (rank == rank0) {
      for (rank1 = 0; rank1<numtasks; rank1++) {
        if (rank1!=rank0){
          dest=rank1;
          source=rank1;
          printf("Rank0=%d, From: %d, To=%d\n", rank0, rank0, rank1);
          rc = MPI_Isend(&msg, 1, MPI_CHAR, rank1, tag, MPI_COMM_WORLD, &rq);
          rc = MPI_Irecv(&msg, 1, MPI_CHAR, rank1, tag, MPI_COMM_WORLD, &rq);
          printf("Rank1=%d, From: %d, To=%d\n", rank1, rank1, rank0);
          rc = MPI_Isend(&msg, 1, MPI_CHAR, rank0, tag, MPI_COMM_WORLD, &rq);
          rc = MPI_Irecv(&msg, 1, MPI_CHAR, rank0, tag, MPI_COMM_WORLD, &rq);
          }
        }
     }
  }

//   for (i=2; i<3; i++) {
//     int n_char = pow(2,i);
//     char msg = 'x';//[2^i];
//     // char msg[n_char];
//     // memset(msg, 'x', n_char*sizeof(char));
//     int chunksize = sizeof(msg);
//     if (rank == rank0) {
//       for (to=0; to<numtasks; to++){
//         if(to!=rank0){
//           printf("From: %d, To=%d\n", rank0 ,to);
//           dest = to;
//           source = to;
//           // Initialize MPI clock
//           Tstart = MPI_Wtime();
//           rc = MPI_Isend(&msg, chunksize, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &rq);
//           rc = MPI_Irecv(&msg, chunksize, MPI_CHAR, source, tag, MPI_COMM_WORLD, &rq);
//           Tend = MPI_Wtime();
//           delT = Tend - Tstart;
//           sumT+=delT*1000000;
//           tarray[n]=delT*1000000;
//           }
//         }
//          for (n=0; n<=reps; n++)
//            printf("%f\n",tarray[n]);
//
//          avgT = (sumT)/reps;
//          stdev = std(tarray, reps);
//          printf("%d %0.2f %0.2f\n",8*n_char, avgT, stdev);
//         }
//     else if (rank!=rank0) {
//        dest = rank0;
//        source = rank0;
//       //  printf("From: %d, To=%d\n", rank,rank0);
//        rc = MPI_Irecv(&msg, chunksize, MPI_CHAR, source, tag, MPI_COMM_WORLD, &rq);
//        rc = MPI_Isend(&msg, chunksize, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &rq);
//     }
//   }
// }
MPI_Finalize();
exit(0);
}

float std(float data[], int n) {
    float mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/n);
}
