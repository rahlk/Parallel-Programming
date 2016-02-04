#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void cudaADD(int* a, int* b) {
  a[0]+=b[0];
}

int main(){
  int a=5, b=6;
  int *c_a, *c_b;

  // Allocate memory for CUDA
  cudaMalloc(&c_b, sizeof(int));
  cudaMalloc(&c_a, sizeof(int));

  // Transfer data to GPU from CPU
  cudaMemcpy(c_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_b, &b, sizeof(int), cudaMemcpyHostToDevice);

  // Run on GPU
  cudaADD<<<1,1>>>(c_a, c_b);

  // Transfer for GPU to CPU
  cudaMemcpy(&a, c_a, sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d\n", a);

  // Free allocated memory
  cudaFree(c_a);
  cudaFree(c_b);

  return 0;
}
