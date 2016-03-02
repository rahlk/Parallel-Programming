#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include <stdbool.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG
#define VSQR 0.1
#define TSCALE 1.0
#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0
#define USE_MATH_DEFINES
#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1
#define NINEPTSTENCIL 1
#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, int rank, double *row, double *col, double *indv, double h, double dt, double t);
void evolve9pt_1(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);
void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);
void transfer(double *from, double *to, int r, int n, bool dir);
void dest(double *source, double *row, double *col, double *indv, int *hor, int *ver, int *diag, int rank, int size);
extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads);


/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

__device__ double f_CUDA(double p, double t)
{
  return -__expf(-TSCALE * t) * p;
}

__global__ void evolve9ptCUDA(double *un, double *uc, double *uo, double *pebbles, int n, int rank, double *row, double *col, double *indv, double h, double dt, double t) {
  int idx = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
  int i = idx / n;
  int j = idx % n;
  if(!(i == 0 || i == n - 1 || j == 0 || j == n - 1)) {
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25*(uc[idx + n - 1] + uc[idx + n + 1] + uc[idx - n - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
  }
  else {
    // Code for the fringe regions goes here...
    switch (rank) {
      case 1:
        if (i==0 || j==0) {
          un[idx]=0.;
        }
        else if(i==n-1 && j==n-1) { // Bottom right corner
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + col[i] + row[j] + uc[idx - n] + 0.25*(row[j-1] + *indv + uc[idx - n - 1] + col[i-1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(j==n-1 && i<n-1) { // Right edge
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + col[i] + uc[idx + n] + uc[idx - n] + 0.25*(uc[idx + n - 1] + col[i + 1] + uc[idx - n - 1] + col[i-1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(i==n-1 && j<n-1) { // Bottom edge
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + row[j] + uc[idx - n] + 0.25*(row[j - 1] + row[j + 1] + uc[idx - n - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        break;

      case 2:
        if (i==0 || j==n-1){
          un[idx]=0.;
        }
        else if(i==n-1 && j==0) { // Bottom left corner
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((col[i] + uc[idx+1] + row[j] + uc[idx - n] + 0.25*(*indv + row[j + 1] + col[i - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(j==0 && i<n-1) { // Left edge
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((col[i] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25*(col[i+1] + uc[idx + n + 1] + col[i-1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(i==n-1 && j>0) { // Bottom Edge
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + row[j] + uc[idx - n] + 0.25*(row[j - 1] + row[j + 1] + uc[idx - n - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        break;

      case 3:
        if (i==n-1 || j==0){
          un[idx]=0.;
        }
        else if(i==0 && j==n-1) { // Top right corner
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + col[i] + uc[idx + n] + row[j] + 0.25*(uc[idx + n - 1] + col[i+1] + row[j-1] + *indv)- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(j==n-1 && i>0) { // Left edge
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + col[i] + uc[idx + n] + uc[idx - n] + 0.25*(uc[idx + n - 1] + col[i+1] + uc[idx - n - 1] + col[i-1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(i==0 && j<n-1) { // Top Edge
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + row[j] + 0.25*(uc[idx + n - 1] + uc[idx + n + 1] + row[j-1] + row[j+ 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        break;

      case 4:
        if (i==n-1 || j==n-1){
          un[idx]=0.;
        }
        else if(i==0 && j==0) { // Top left corner
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((col[i] + uc[idx+1] + uc[idx + n] + row[j] + 0.25*(col[i+1] + uc[idx+n+1] + *indv + row[j+1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(j==0 && i>0) { // Right edge
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((col[i] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25*(col[i+1] + uc[idx + n + 1] + col[i-1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        else if(i==0 && j>0) { // Top Edge
          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + row[j] + 0.25*(uc[idx + n - 1] + uc[idx + n + 1] + row[j-1] + row[j+1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
        }
        break;
    }
  }}


int main(int argc, char *argv[])
{

  int   numproc, rank;

  MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status *status;
  MPI_Request *request;
  request = (MPI_Request *) malloc(numproc * sizeof(MPI_Request));
  status = (MPI_Status *) malloc(numproc * sizeof(MPI_Status));

  int     npoints   = 256;// atoi(argv[1]);
  int     npebs     = 3;// atoi(argv[2]);
  double  end_time  = 1.00;// (double)atof(argv[3]);
  int     nthreads  = 1024;// atoi(argv[4]);
  int 	  narea	    = npoints * npoints;
  bool    once      = true;
  int size=(npoints/2)*(npoints/2);

  double t, dt;
  double h = (XMAX - XMIN)/npoints;

  if (rank == 0) {

    double *u_i0, *u_i1;
    double *u_cpu, *u_gpu, *pebs;
    double *peb, *n_cpu; //1, *n_cpu2, *n_cpu3, *n_cpu4;
    double elapsed_cpu, elapsed_gpu;
    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

    peb   = (double*)malloc(sizeof(double) * size);
    u_i0  = (double*)malloc(sizeof(double) * narea);
    u_i1  = (double*)malloc(sizeof(double) * narea);
    u_cpu = (double*)malloc(sizeof(double) * narea);
    u_gpu = (double*)malloc(sizeof(double) * narea);
    n_cpu = (double*)malloc(sizeof(double) * size);

    pebs = (double*)malloc(sizeof(double) * narea);
    printf("Rank0: Running a (%d x %d) grid, until %f, with %d threads\n", npoints, npoints, end_time, nthreads);
    init_pebbles(pebs, npebs, npoints);
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);

    // Initial
    run_cpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time);
    print_heatmap("lake_cpu_f.dat", u_gpu, npoints, h);

    // Tranfer to MPI nodes
    int i;
    for (i=1; i<numproc; i++) {
      transfer(pebs, peb, i, npoints, true); //get corresponding data
      MPI_Send(peb,size, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
      }

      /*-----------------------------------*/
      /* Stitch individual nodes together */
      /*
      MPI_Recv(n_cpu, size, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD, &status[1]);
      transfer(n_cpu, u_cpu, 1, npoints, false);

      MPI_Recv(n_cpu, size, MPI_DOUBLE, 2, 12, MPI_COMM_WORLD, &status[2]);
      transfer(n_cpu, u_cpu, 2, npoints, false);

      MPI_Recv(n_cpu, size, MPI_DOUBLE, 3, 13, MPI_COMM_WORLD, &status[3]);
      transfer(n_cpu, u_cpu, 3, npoints, false);

      MPI_Recv(n_cpu, size, MPI_DOUBLE, 4, 14, MPI_COMM_WORLD, &status[4]);
      transfer(n_cpu, u_cpu, 4, npoints, false);

      // Save final Image
      print_heatmap("lake_cpu_mpi.dat", u_cpu, npoints, h);
      */
      /*-----------------------------------*/


    }

  else {
      /* For Reference:

      + : Fringe edges
      X : Diagonal fringe point

      |``````````````````+|+````````````````|
      |                  +|+                |
      |                  +|+                |
      |     Rank 1       +|+     Rank 2     |
      |                  +|+                |
      |++++++++++++++++++X|X++++++++++++++++|
      |++++++++++++++++++X|X++++++++++++++++|
      |                  +|+                |
      |                  +|+                |
      |     Rank 3       +|+     Rank 4     |
      |                  +|+                |
      |                  +|+                |
      |                  +|+                |
      ```````````````````````````````````````
      */

    cudaEvent_t kstart, kstop;
    float ktime;

    int number_amount;
    double *un  , *u0  , *u1  , *uc      , *uo, *pebble;
    double *d_un, *d_uc, *d_uo, *d_pebble, *d_temp;
    int n = npoints/2;

    u0     = (double*)malloc(sizeof(double) * n*n);
    u1     = (double*)malloc(sizeof(double) * n*n);
    un     = (double*)malloc(sizeof(double) * n*n);
    uc     = (double*)malloc(sizeof(double) * n*n);
    uo     = (double*)malloc(sizeof(double) * n*n);
    pebble = (double*)malloc(sizeof(double) * n*n);

    MPI_Recv(pebble, size, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /*-----------------------------------*/
    /* Sanity Check!*/ /*
    MPI_Get_count(&status, MPI_INT, &number_amount);
    printf("1 received %d numbers from 0. Message source = %d, "
           "tag = %d\n",
           number_amount, status.MPI_SOURCE, status.MPI_TAG);
    /*-----------------------------------*/

    init(u0, pebble, npoints/2);
    init(u1, pebble, npoints/2);

    // Begin Timer
    t = 0.;
    dt = h / 2.;

    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaEventCreate(&kstart));
    CUDA_CALL(cudaEventCreate(&kstop));

    cudaMalloc((void **)&d_un, sizeof(double) * n * n);
    cudaMalloc((void **)&d_uc, sizeof(double) * n * n);
    cudaMalloc((void **)&d_uo, sizeof(double) * n * n);
    cudaMalloc((void **)&d_pebble, sizeof(double) * n * n);

    cudaMemcpy(d_uo, u0,    sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uc, u1,    sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pebble, pebble, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    dim3 block_dim(nthreads, nthreads,1);
    dim3 grid_dim(n/nthreads, n/nthreads,1);

    /* Start GPU computation timer */
    CUDA_CALL(cudaEventRecord(kstart, 0));

    while(1) {
      // What to send where
      double *row, *col, *indv;
      row = (double*)malloc(sizeof(double) * npoints/2);
      col = (double*)malloc(sizeof(double) * npoints/2);
      indv = (double*)malloc(sizeof(double));

      int *hor, *ver, *diag;

      hor  = (int*)malloc(sizeof(int));
      ver  = (int*)malloc(sizeof(int));
      diag = (int*)malloc(sizeof(int));

      dest(un, row, col, indv, hor, ver, diag, rank, npoints/2);

      // Send boundaries to respective neighbours
      MPI_Send(row , npoints/2, MPI_DOUBLE, *ver , rank, MPI_COMM_WORLD);
      MPI_Send(col , npoints/2, MPI_DOUBLE, *hor , rank, MPI_COMM_WORLD);
      MPI_Send(indv, 1,         MPI_DOUBLE, *diag, rank, MPI_COMM_WORLD);

      // Compute turbulance: Receive neighbours
      MPI_Recv(row,  npoints/2, MPI_DOUBLE, *hor,  *hor,  MPI_COMM_WORLD, &status[rank]);
      MPI_Recv(col,  npoints/2, MPI_DOUBLE, *ver,  *ver,  MPI_COMM_WORLD, &status[rank]);
      MPI_Recv(indv, 1, MPI_DOUBLE, *diag, *diag, MPI_COMM_WORLD, &status[rank]);

      // Nine point stencil on CUDA cores
      evolve9ptCUDA<<<grid_dim, block_dim>>>(d_un, d_uc, d_uo, d_pebble, n, rank, row, col, indv, h, dt, t);
      d_temp = d_uc;
      d_uc = d_un;
      d_un = d_uo;
      d_uo = d_temp;

      if(!tpdt(&t,dt,end_time)) {
        break;
      }
    }

    cudaMemcpy(un, d_un, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    /*-----------------------------------*/
    // // Send final results to Rank 0.
    // MPI_Isend(un, size, MPI_DOUBLE, 0, rank+10,  MPI_COMM_WORLD, &request[0]);
    // printf("Done\n");
    /*-----------------------------------*/

    // Initial Output files
    char* s;
    s = (char*)malloc(sizeof(char)*17);
    int k = sprintf(s, "lake_node_%d.dat", rank);
    if (k>=0)
      print_heatmap(s, un, npoints/2, h);
    else {
      printf("Error in filename!\n");
      MPI_Finalize();
      return 0;
    }
  }

  MPI_Finalize();
  return 0;
}

void dest(double *source, double *row, double *col, double *indv, int *hor, int *ver, int *diag, int myrank, int size) {
  int i, x, y;
  switch (myrank) {
    case 1:
      for(i=0; i<size; i++) {
        x=size*(size-1)+i;
        y=i*size+(size-1);
        row[i]=source[x];
        col[i]=source[y];
      }
      *indv = source[(size-1)*(size-1)-1];
      *ver  = 2;
      *diag = 4;
      *hor  = 3;
      break;
    case 2:
      for(i=0; i<size; i++) {
        x=size*(size-1)+i;
        y=i;
        row[i]=source[x];
        col[i]=source[i];
      }
      *indv = source[(size-1)*(size-1)-1];
      *ver  = 1;
      *diag = 3;
      *hor  = 4;
      break;
    case 3:
      for(i=0; i<size; i++) {
        x=i;
        y=i*size+(size-1);
        row[i]=source[x];
        col[i]=source[y];
      }
      *indv = source[(size-1)*(size-1)-1];
      *ver  = 4;
      *diag = 2;
      *hor  = 1;
      break;
    case 4:
      for(i=0; i<size; i++) {
        x=i;
        y=i;
        row[i]=source[x];
        col[i]=source[y];
      }
      *indv = source[(size-1)*(size-1)-1];
      *ver  = 3;
      *diag = 1;
      *hor  = 2;
      break;
  }
}
void transfer(double *from, double *to, int r, int n, bool dir) {
  // This is really naive. I'll probably change it.
  int x,y, idx_t, idx_f;
  for (x=0; x<(int) n/2; x++)
  for (y=0; y<(int) n/2; y++) {
    if (r==1) {
      idx_t = x*n/2+y;
      idx_f = x*n+y;
    }
    else if (r==2) {
      idx_t = x*n/2+y;
      idx_f = x*n+n/2+y;
    }
    else if (r==3) {
      idx_t=x*n/2+y;
      idx_f=(x+n/2)*n+y;
    }
    else if (r==4) {
      idx_t=x*n/2+y;
      idx_f=(x+n/2)*n+y+n/2;
    }
    if (dir==true)
      to[idx_t]=from[idx_f];
    else
      to[idx_f]=from[idx_t];
  }
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  double *un, *uc, *uo;
  double t, dt;

  un = (double*)malloc(sizeof(double) * n * n);
  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  while(1)
  {

    evolve9pt_1(un, uc, uo, pebbles, n, h, dt, t);
    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * n * n);
}

void evolve9pt_1(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25*(uc[idx + n - 1] + uc[idx + n + 1] + uc[idx - n - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}


void print_heatmap(char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %0.2e\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}
