#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <mpi.h>

#define __DEBUG
#define VSQR 0.1
#define TSCALE 1.0
#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)


extern int tpdt(double *t, double dt, double end_time);

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

__global__ void evolve9ptCUDA(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t) {
  int idx = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
  int i = idx / n;
  int j = idx % n;
  if(!(i == 0 || i == n - 1 || j == 0 || j == n - 1))
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25*(uc[idx + n - 1] + uc[idx + n + 1] + uc[idx - n - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
  else un[idx] = 0.;
}
__global__ void evolve9ptCUDA_mpi(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int startRow) {
  int idx = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
  int i = idx / n;
  int j = idx % n;
  idx = idx + startRow*n;
  if(!(i == 0 || i == n - 1 || j == 0 || j == n - 1))
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25*(uc[idx + n - 1] + uc[idx + n + 1] + uc[idx - n - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
  else un[idx] = 0.;
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
        
	double *un, *uc, *uo, *pb, *temp;
  double t, dt;
        /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

  t = 0.;
  dt = h/2.;

  cudaMalloc((void **)&un, sizeof(double) * n * n);
  cudaMalloc((void **)&uc, sizeof(double) * n * n);
  cudaMalloc((void **)&uo, sizeof(double) * n * n);
  cudaMalloc((void **)&pb, sizeof(double) * n * n);

  cudaMemcpy(uo, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(uc, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(pb, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);

  dim3 block_dim(nthreads, nthreads,1);
  dim3 grid_dim(n/nthreads, n/nthreads,1);

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));
  while(1)
  {
    evolve9ptCUDA<<<grid_dim, block_dim>>>(un, uc, uo, pb, n, h, dt, t);
    temp = uc;
    uc = un;
    un = uo;
    uo = temp;
    if(!tpdt(&t, dt, end_time))
      break;
  }
	cudaMemcpy(u, uc, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
        /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	cudaFree(un);
  cudaFree(uc);
  cudaFree(uo);
  cudaFree(pb);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
void run_gpu_mpi(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int numproc, int rank)
{
        
  int startRow, endRow, rowsPerPart, totalElements;
  rowsPerPart = n/numproc;
  startRow = rowsPerPart * rank;
  endRow = rowsPerPart * (rank + 1);
  double *un, *uc, *uo, *pb, *temp;
  double t, dt;

  t = 0.;
  dt = h/2.;

  double *uc_host, *uo_host, *un_host;
  uc_host=(double *)malloc(sizeof(double)*n*n);
  uo_host=(double *)malloc(sizeof(double)*n*n);
  un_host=(double *)malloc(sizeof(double)*n*n);
  memset(un_host, 0, sizeof(double)*n*n);

  cudaMalloc((void **)&un, sizeof(double) * n * n);
  cudaMalloc((void **)&uc, sizeof(double) * n * n);
  cudaMalloc((void **)&uo, sizeof(double) * n * n);
  cudaMalloc((void **)&pb, sizeof(double) * n * n);

  cudaMemcpy(uo, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(uc, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(un, un_host, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(pb, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);

  dim3 block_dim(nthreads, nthreads,1);
  dim3 grid_dim(n/(nthreads * 2), n/(nthreads * 2),1);

    int iter = 0;
  while(1)
  {
    MPI_Request *requestList;
    int reqCount = 0;
    requestList = (MPI_Request *)malloc(sizeof(MPI_Request)*8);
    evolve9ptCUDA_mpi<<<grid_dim, block_dim>>>(un, uc, uo, pb, n, h, dt, t, startRow);
    if(!tpdt(&t, dt, end_time))
      break;
    printf("%d iteration in rank %d with startRow as %d.\n", iter, rank, startRow);
    iter++;
    cudaMemcpy(uc_host, uc, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(un_host, un, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    printf("Copied from device.\n");
    if(rank!=0)
    {
      MPI_Irecv(un_host + (rowsPerPart * rank - 1)*n , n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &(requestList[reqCount]));
      MPI_Irecv(uc_host + (rowsPerPart * rank - 1)*n , n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &(requestList[reqCount+1]));
      reqCount = reqCount + 2;
    }
    if(rank != numproc - 1)
    {
      MPI_Irecv(un_host + rowsPerPart * (rank + 1) * n, n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requestList[reqCount]);
      MPI_Irecv(uc_host + rowsPerPart * (rank + 1) * n, n, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requestList[reqCount+1]);
      reqCount = reqCount + 2;
    }
    if(rank != 0)
    { 
      MPI_Isend(un_host + (rowsPerPart * rank)*n, n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requestList[reqCount]);
      MPI_Isend(uc_host + (rowsPerPart * rank)*n, n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requestList[reqCount+1]);
      reqCount = reqCount + 2;
    }
    if(rank != numproc-1)
    {
      MPI_Isend(un_host + (rowsPerPart * (rank + 1) - 1)*n, n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requestList[reqCount]);
      MPI_Isend(uc_host + (rowsPerPart * (rank + 1) - 1)*n, n, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requestList[reqCount+1]);
      reqCount = reqCount + 2;
    }
    MPI_Waitall(reqCount, requestList, MPI_STATUS_IGNORE);
    cudaMemcpy(uo, uc_host, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(uc, un_host, sizeof(double)*n*n, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(u, un, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

  cudaFree(un);
  cudaFree(uc);
  cudaFree(uo);
  cudaFree(pb);
}
