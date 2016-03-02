#include <stdio.h>
#include <stdlib.h>
#include "sock_msg.h"
#define MAX_MSG_SIZE 16
#define MAX_PROC 16

int main(int argc, char **argv) {
  int port=atoi(argv[1]);
  int s = dsm_server_open(SOCK_STREAM, &(port), FALSE, NULL, 0);
  printf("%d\n",s);
  return 0;
}
