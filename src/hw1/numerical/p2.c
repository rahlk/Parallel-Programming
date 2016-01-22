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
#define  NGRID 10000
/* first grid point */
#define  XI  1.0
/* last grid point */
#define  XF  100.0

/* floating point precision type definitions */
typedef  double  FP_PREC;

/* function declarations */
FP_PREC fn(FP_PREC);
FP_PREC dfn(FP_PREC);
FP_PREC ifn(FP_PREC, FP_PREC);
void print_function_data(int, FP_PREC*, FP_PREC*, FP_PREC*);
void print_error_data(int np, FP_PREC, FP_PREC, FP_PREC*, FP_PREC*, FP_PREC);
int main(int, char**);

int main (int argc, char *argv[])
{
    int  numproc, rank, len,i;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);
    
    FP_PREC *yc, *dyc, *derr, *fullerr;
    FP_PREC *xc, dx, intg, davg_err, dstd_dev, intg_err;
    FP_PREC globalSum = 0.0;
    
    //"real" grid indices
    int imin, imax;
    
    imin = 1 + (rank * (NGRID/numproc));
    
    if(rank == numproc - 1)
    imax = NGRID;
    
    else
    imax = (rank+1) * (NGRID/numproc);
    
    
    int range = imax - imin + 1;
    
    xc =  (FP_PREC*) malloc((range + 2) * sizeof(FP_PREC));
    yc =  (FP_PREC*) malloc((range + 2) * sizeof(FP_PREC));
    dyc =  (FP_PREC*) malloc((range + 2) * sizeof(FP_PREC));
    dx = (XF - XI)/(double)NGRID;
    for (i = 1; i <= range ; i++)
    {
        //xc[i] = imin + (XF - XI) * (FP_PREC)(i - 1)/(FP_PREC)(NGRID - 1);
        xc[i] = XI + dx * (imin + i - 2);
    }
    
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
        intg += 0.5 * (xc[i + 1] - xc[i]) * (yc[i + 1] + yc[i]);
    }
    
    MPI_Reduce(&intg, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    
    //compute the error, average error of the derivatives
    derr = (FP_PREC*)malloc((range + 2) * sizeof(FP_PREC));
    
    //compute the errors
    for(i = 1; i <= range; i++)
    {
        derr[i] = fabs((dyc[i] - dfn(xc[i]))/dfn(xc[i]));
    }
    
    derr[0] = derr[range + 1] = 0.0;
    
    if(rank == 0)
    {
        fullerr = (FP_PREC *)malloc(sizeof(FP_PREC)*NGRID);
        for(i = 0;i<range;i++)
        {
            fullerr[i] = derr[i+1];
        }
        for(i = 1; i<numproc; i++)
        {
            int rmin, rmax;
            
            rmin = 1 + (i * (NGRID/numproc));
            
            if(i == numproc - 1)
            rmax = NGRID;
            else
            rmax = (i+1) * (NGRID/numproc);
            MPI_Recv(fullerr+rmin-1, rmax-rmin+1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        double sum = 0.0;
        for(i=0; i<NGRID; i++)
        {
            sum+=fullerr[i];
        }
        davg_err = sum/(FP_PREC)NGRID;
        dstd_dev = 0.0;
        for(i = 0; i< NGRID; i++)
        {
            dstd_dev += pow(derr[i] - davg_err, 2);
        }
        dstd_dev = sqrt(dstd_dev/(FP_PREC)NGRID);
        
        intg_err = fabs((ifn(XI, XF) - globalSum)/ifn(XI, XF));
        printf("%e: %e: %en", davg_err, dstd_dev, intg_err);
    }
    else
    {
        MPI_Send(derr+1, imax-imin+1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        fflush(stdout);
    }
    
    MPI_Finalize();
}
