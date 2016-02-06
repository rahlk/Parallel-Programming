#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1
#define NINEPTSTENCIL 1

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

void transfer(double *from, double *to, int x_off, int y_off);
extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads);

int main(int argc, char *argv[])
{

  if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }


  int   numproc, rank;

  MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int     npoints   = 256;// atoi(argv[1]);
  int     npebs     = 3;// atoi(argv[2]);
  double  end_time  = 1.00;// (double)atof(argv[3]);
  int     nthreads  = 1024;// atoi(argv[4]);
  int 	  narea	    = npoints * npoints;
  bool    once      = TRUE;
  int size=(npoints/2+1)*(npoints/2+1);
  if (rank == 0) {

    double *u_i0, *u_i1;
    double *u_cpu, *u_gpu, *pebs;
    double *node_1,*node_2,*node_3,*node_4;
    double h;

    double elapsed_cpu, elapsed_gpu;
    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

    node_1 = (double*)malloc(sizeof(double) * (npoints/2+1)*(npoints/2+1));
    node_2 = (double*)malloc(sizeof(double) * (npoints/2+1)*(npoints/2+1));
    node_3 = (double*)malloc(sizeof(double) * (npoints/2+1)*(npoints/2+1));
    node_4 = (double*)malloc(sizeof(double) * (npoints/2+1)*(npoints/2+1));

    if(once) {
      u_i0 = (double*)malloc(sizeof(double) * narea);
      u_i1 = (double*)malloc(sizeof(double) * narea);
      pebs = (double*)malloc(sizeof(double) * narea);
      printf("Rank0: Running a (%d x %d) grid, until %f, with %d threads\n", npoints, npoints, end_time, nthreads);
      h = (XMAX - XMIN)/npoints;
      init_pebbles(pebs, npebs, npoints);
      init(u_i0, pebs, npoints);
      init(u_i1, pebs, npoints);
      print_heatmap("lake_i.dat", u_i0, npoints, h);
      // Copy initial lake to 4 nodes
      transfer(pebs, node_1, 1, npoints);
      transfer(pebs, node_2, 2, npoints);
      transfer(pebs, node_3, 3, npoints);
      transfer(pebs, node_4, 4, npoints);
      // Tranfer to MPI nodes
      MPI_Send(node_1,size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
      MPI_Send(node_2,size, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD);
      MPI_Send(node_3,size, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD);
      MPI_Send(node_4,size, MPI_DOUBLE, 4, 4, MPI_COMM_WORLD);
      // Turn flag off
      once=FALSE;
    }
  }

  else {
    MPI_Status status;
    int number_amount;
    patch = (double*)malloc(sizeof(double) * (npoints/2+1)*(npoints/2+1));
    MPI_Recv(patch,size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &number_amount);
    printf("1 received %d numbers from 0. Message source = %d, "
           "tag = %d\n",
           number_amount, status.MPI_SOURCE, status.MPI_TAG);
  }
  MPI_Finalize();

}

void transfer(double *from, double *to, int rank, int n) {
  // This is really stupid. I'll change it...
  int x,y, idx_t, idx_f;
  for (x=0; x<=(int) n/2; x++)
  for (y=0; y<=(int) n/2; y++) {
    switch(rank) {
      case 1:
      idx_t = x*n+y;
      idx_f = x*n+y;
      to[idx_t]=from[idx_f];
      case 2:
      idx_t = x*n+y;
      idx_f = x*n+n/2+y-1;
      to[idx_t]=from[idx_f];
      case 3:
      idx_t=x*n+y;
      idx_f=(x+1)*n+y;
      to[idx_t]=from[idx_f];
      case 4:
      idx_t=x*n+y;
      idx_f=(x+3/2)*n+y-1;
      to[idx_t]=from[idx_f];
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
    if(NINEPTSTENCIL)
    {
      evolve9pt(un, uc, uo, pebbles, n, h, dt, t);
    }
    else
    {
      evolve(un, uc, uo, pebbles, n, h, dt, t);
    }
    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * n * n);
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

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
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
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
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
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}
