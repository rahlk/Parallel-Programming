/*
rkrish11 Rahul Krishna
*/
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define SEED 35791246



__global__ void init_stuff(curandState *state, int count) {
// This sets a random number seed for all the threads
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 if (idx<count)
  curand_init(1337, idx, 0, &state[idx]);
}


__global__ void cudaMonte(double* pi, int count, curandState* state) {

  // Perfome MC simulation on the threads
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  double x,y,z;

  if (id<count) {
    x = (double)curand_uniform(&state[id]);
    y = (double)curand_uniform(&state[id]);
    z = x*x+y*y;
    if (z<=1) pi[id]=1;
    else pi[id]=0;
  }
  __syncthreads();

  // Find the total number of points that lie inside the quadrant of the cirle
  for (int i=1; i<count;i++) {
    pi[0]+=pi[i];
  }
}

int main(int argc, char** argv) {
  int niter=0;
  double pi;
  double* d_pi;
  curandState *d_state;

  printf("Enter the number of iterations used to estimate pi: ");
  scanf("%d",&niter);

  double* h_pi = new double[niter];

  if (cudaMalloc(&d_pi, sizeof(int)*niter) != cudaSuccess) {
      printf("Error in memory allocation.\n");
      return 0;
  }
  if (cudaMalloc(&d_state, sizeof(curandState)*niter) != cudaSuccess) {
      printf("Error in memory allocation for random state.\n");
      return 0;
  }
  if (cudaMemcpy (d_pi, h_pi, sizeof(int)*niter, cudaMemcpyHostToDevice) != cudaSuccess) {
      printf("Error in copy from host to device.\n");
      cudaFree(d_pi);
      return 0;
  }

  // Number of threads = 1024, number of blocks = (int) (niter/threads)+1
  init_stuff<<<(int) niter/1024+1, 1024>>>(d_state, niter);
  cudaMonte<<<(int) niter/1024+1, 1024>>>(d_pi, niter, d_state);

  if (cudaMemcpy (h_pi, d_pi, sizeof(int)*niter, cudaMemcpyDeviceToHost) != cudaSuccess) {
      printf("Error in copy from device to host.\n");
      delete[] h_pi;
      cudaFree(d_pi);
      return 0;
  }

  // Final Estimate of pi
  pi= (double) h_pi[0]/niter*4;
  printf("# of trials= %d , estimate of pi is %g \n",niter,pi);

}
