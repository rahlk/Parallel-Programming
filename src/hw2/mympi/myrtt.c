#include <stdio.h>
#include <stdlib.h>
#include "mympi.h"
#define MAX_MSG_SIZE 16
#define MAX_PROC 16


int main(int argc, char **argv) {
  if(argc<3) {
    printf("Insufficient arguents, Reveived %d argument(s). Usage %s <rank> <n_proc>\n", argc - 1, argv[0]);
    exit(0);
  }
  char *msg;
  msg=(char *)malloc(sizeof(char)*MAX_MSG_SIZE);
  memset(msg, 'a', sizeof(char)*MAX_MSG_SIZE);
  int rank, numproc;
  int *handle=(int)malloc(sizeof(int));
  int *status=(int)malloc(sizeof(int));

  mympi_init(argc, argv, &rank, &numproc, handle);
  char *recvd = (char*)malloc(sizeof(char));
  // // Send data
  // printf("%d\n", *(handle+rank));
  // int i=0;
  //
  if(rank==0) {
    // for(i=0;i<numproc;i++) printf("%d\n", *(handle+i));
    mympi_send(msg, 1, 1, 0, *(handle+rank), status);
  }
  else {
    // for(i=0;i<numproc;i++) printf("%d\n", *(handle+i));
    mympi_recv(recvd, 1, 0, rank, *(handle+rank), status);
    printf("Rank:%d: Msg:%s\n", rank, recvd);
  }

  // mympi_send(msg, 2, 1, 0, *(handle+rank), status);
  // mympi_recv(recvd, 2, 0, rank, *(handle+rank+1), status);
  // printf("Rank:%d: Msg:%s\n", rank, recvd);
  // mympi_close(*(handle+rank));
  return(0);
}


/* Junk

/* Sanity check -------
printf("Rank:%d :: Numproc:%d\n\nHandles\n", *rank, *numproc);
int i;
for (i=0; i<*numproc; i++) {
  printf("%d\n", handle[i]);
}
----------------------- */
