#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sock_msg.h"

// char** readNodes();
typedef enum {
  mympi_Char,
  mympi_Int,
  mympi_Float,
  mympi_Double,
} mympi_Type;

int client(char *msg, char *dsthost, int dstport);

int server();

int mympi_init(int argc, char **argv, int* rank, int* numproc, int* handle);

void mympi_close(int handle);

int mympi_send(char *msg, int blocklen, int dest, int tag, int handle, int *status);

int mympi_recv(char *msg, int blocklen, int dest, int tag, int handle, int *status);

char **fileAsArray(char *filename);
