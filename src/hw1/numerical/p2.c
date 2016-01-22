/******************************************************************************
* FILE: hw1.c
* DESCRIPTION:
*
* Users will supply the functions
* i.) fn(x) - the function to be analyized
* ii.) dfn(x) - the true derivative of the function
* iii.) ifn(x) - the true integral of the function
*
* The function fn(x) should be smooth and continuous, and
* the derivative and integral should have analyitic expressions
* on the entire domain.
*
* AUTHOR: Christopher Mauney
* LAST REVISED: 08/18/12
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

/* The number of grid points */
#define   NGRID           100
/* first grid point */
#define   XI              1.0
/* last grid point */
#define   XF              100.0

/* floating point precision type definitions */
typedef   double   FP_PREC;

/* function declarations */
FP_PREC     fn(FP_PREC);
FP_PREC     dfn(FP_PREC); 
FP_PREC     ifn(FP_PREC, FP_PREC);
void        print_function_data(int, FP_PREC*, FP_PREC*, FP_PREC*);
void        print_error_data(int np, FP_PREC, FP_PREC, FP_PREC*, FP_PREC*, FP_PREC);
int         main(int, char**);

int main (int argc, char *argv[])
{
	int   numproc, rank, len,i;
	char  hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(hostname, &len);
	
    FP_PREC     *yc, *dyc;
	FP_PREC     *xc, dx, intg;

    //"real" grid indices
    int         imin, imax;  

    imin = 1 + (rank * (NGRID/numproc));
	
	if(rank == numproc - 1)
		imax = XF;
	
	else
    imax = (rank+1) * (NGRID/numproc);
	
	printf("min: %d, max: %d\n",imin,imax);
	
	int range = imax - imin + 1;
	
	xc  =   (FP_PREC*) malloc((range + 2) * sizeof(FP_PREC));
	yc  =   (FP_PREC*) malloc((range + 2) * sizeof(FP_PREC));
	dyc  =   (FP_PREC*) malloc((range + 2) * sizeof(FP_PREC));
	
    for (i = 1; i <= range ; i++)
    {
      xc[i] = imin + (XF - XI) * (FP_PREC)(i - 1)/(FP_PREC)(NGRID - 1);
    }
	
    dx = xc[2] - xc[1];
    xc[0] = xc[1] - dx;
    xc[range + 1] = xc[range] + dx;
	
    for( i = 1; i <= range; i++ )
    {
      yc[i] = fn(xc[i]);
    }
	
	yc[0] = fn(xc[0]);
	yc[range + 1] = fn(xc[range + 1]);
	
    for (i = 1; i <= range; i++)
    {
      dyc[i] = (yc[i + 1] - yc[i - 1])/(2.0 * dx);
    }
	
    intg = 0.0;
    for (i = 1; i <= range; i++)
    {
      //there are NGRID points, so there are NGRID-1 integration zones.	
		if((rank == numproc - 1) && (i == range))
		  break;
		  
		intg += 0.5 * (xc[i + 1] - xc[i]) * (yc[i + 1] + yc[i]);
    }
	
	//if(rank == 0)
	{
		for(i=0; i<=range+1; i++)
			printf("%lf: %lf: %lf\n",xc[i],yc[i],dyc[i]);
		printf("Integration: %lf\n",intg);
	}
	
	MPI_Finalize();
}
