#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sock_msg.h"
#define MAX_BUFSZ 4096 /*max text line length*/
#define SERV_START_PORT 4096 /*port*/
#include <unistd.h>
// char** readNodes();
typedef enum {
  mympi_Char,
  mympi_Int,
  mympi_Float,
  mympi_Double,
} mympi_Type;

int client(char *msg, char *dsthost, int dstport) {
  // char msg[256];
  int s;
  // int dstport, char *dsthost;
  /* compose message, determine destination host/port */
  // memset(msg, 0, sizeof(msg));
  printf("%s\n", msg);
  s = dsm_client_open(SOCK_STREAM, dstport, FALSE, dsthost, NULL);
  dsm_write(s, SOCK_STREAM, msg, strlen(msg) + 1, NULL);
  printf("%d\n", s);
  dsm_close(s);
}

int server() {
  int handle, port=SERV_START_PORT;
  int s, len;
  char msg[256], *dummy;

  handle = dsm_server_open(SOCK_STREAM, &port, FALSE, NULL, 0);
  while (1) {
    printf("%d\n", s);
    s = dsm_server_accept(handle, SOCK_STREAM, port, FALSE, NULL,
                          &len, &dummy, 0);
    len = dsm_read(s, SOCK_STREAM, msg, 256, NULL);
    printf("%s\n", msg);
    dsm_close(s);
  }
  dsm_close(handle);
}

char **fileAsArray(char *filename){
  /*
  Adapted from http://stackoverflow.com/questions/19173442/reading-each-line-of-file-into-array
  */

  int lines=16; // Hardcoded for 16 nodes.
  int line_len=100; // Again, arbitrary

  /*allocate memory for test */
  char** words=(char **)malloc(sizeof(char*)*lines);
  if (words==NULL) {
    fprintf(stderr, "Error: Out of memory!\n");
    free(words);
    exit(1);
  }

  FILE *fp = fopen(filename, "r");
  if (fp==NULL) {
    fprintf(stderr, "Error: File not found!\n");
    exit(2);
  }

  int i;
  for (i=0; 1; i++) {
    int j;
    words[i]=malloc(line_len); // Allocate memory for next line.
    if (fgets(words[i],line_len-1,fp)==NULL)
      break;

    /* Get rid of CR or LF at end of line */
    for (j=strlen(words[i])-1;j>=0 && (words[i][j]=='\n' || words[i][j]=='\r');j--);
    words[i][j+1]='\0';
    }
  /* Close file */
  fclose(fp);
  return(words);
}

void mympi_init(int argc, char **argv, int *rank, int *numproc, int* handle) {
  int i, port;
  int s;
  int len=MAX_BUFSZ;
  char *dummy;
  *rank=atoi(argv[1]);
  *numproc=atoi(argv[2]);
  if(realloc(handle, sizeof(int)* *numproc)==NULL) {
    printf("memory alloc failed!\n");
    free(handle);
    exit(0);
  }
  for (i=0; i<*numproc; i++) {
    port=SERV_START_PORT+i;
    // printf("%d\n", port);
    *(handle+i)=dsm_server_open(SOCK_STREAM, &port, FALSE, NULL, 0);
  }
  sleep(2);
}

void mympi_close(int handle) {
  dsm_close(handle);
}

int mympi_send(char *msg, int blocklen, int dest, int tag, int* handle, int* status) {


  // char* toSend;
  char *ackn  = (char*)malloc(sizeof(char));
  char* toSend= (char*)malloc(sizeof(char)*blocklen);
  // printf("%s %d\n", msg, blocklen);
  // toSend = (char *)malloc(sizeof(char));
  int s=-1, s1=-1, len=-1;
  char *dummy;
  //
  // switch(t) {
  //   case mympi_Char:
  //     (char *) msg;
  //     // char * toSend;
  //     toSend = (char *)malloc(sizeof(char)*blocklen);
  //     // break;
  //   case mympi_Int:
  //     (int *) msg;
  //     // int* toSend;
  //     toSend = (int *)malloc(sizeof(int)*blocklen);
  //     // break;
  //   case mympi_Float:
  //     (float *) msg;
  //     // float * toSend;
  //     toSend = (float *)malloc(sizeof(float)*blocklen);
  //   case mympi_Double:
  //     (double *) msg;
  //     // double * toSend;
  //     toSend = (double *)malloc(sizeof(double)*blocklen);
  //     break;
  //   // case default:
  //   //   printf("Error: mympi_Type not recognized.\n");
  //   //   exit(0);
  // }

  // Get "blocklen" chuck of the message
  int i;
  for(i=0; i<blocklen; i++) {
    toSend[i]=msg[i];
  }

  // Get ranks of all the processes
  char **nodes;
  nodes=fileAsArray("NODES");
  // char **ports=fileAsArray("PORTS");
  int port = SERV_START_PORT+dest; // Random Hardcoded port
  printf("%d\n", port);
  /*--------------------------------*/
  /* Insert a failsafe for n>numproc*/
  /*--------------------------------*/

  // while(1) {
    // Send data
    s = dsm_client_open(SOCK_STREAM, port, FALSE, nodes[dest], NULL);
    len = dsm_write(s, SOCK_STREAM, toSend, strlen(toSend), NULL);
    printf("%d\n", len);
    // {
    // Wait for acknowledge
    // printf("%d\n", port+1);
    // s1 = dsm_server_accept(handle, SOCK_STREAM, port+1, FALSE, NULL, &len, &dummy, 0);
    // len = dsm_read(s1, SOCK_STREAM, ackn, 1, NULL);
    // printf("Rank%d: %d %d\n",tag, s1, len);
    // }
    *status=(len>0)?1:0;
    //  dsm_close(s);
    // }
  // dsm_close(s);
  // status=&s;
  // free(toSend);
  return(0);
}

int mympi_recv(char *msg, int blocklen, int src, int tag, int handle, int *status) {

  // Note the variable handle must have the the rank of the receiving node.


  // Get ranks of all the processes
  char **nodes=fileAsArray("NODES");

  // realloc memory for msg
  if(realloc(msg, sizeof(char)*blocklen)==NULL) {
    printf("memory alloc failed!\n");
    exit(0);
  }
  // Get port ids
  // char **ports=fileAsArray("PORTS");

  int port = SERV_START_PORT+tag;
  printf("%d\n", port);
  int s=-1, s1=0, len=-1;
  char *dummy;
  char *ack=(char*)malloc(sizeof(char));
  // // Typecast message
  // switch(t) {
  //   case mympi_Char:
  //     (char *) msg;
  //     msg = (char *)malloc(sizeof(char)*blocklen);
  //   case mympi_Int:
  //     (int *) msg;
  //     msg = (int *)malloc(sizeof(int)*blocklen);
  //   case mympi_Float:
  //     (float *) msg;
  //     msg = (float *)malloc(sizeof(float)*blocklen);
  //   case mympi_Double:
  //     (double *) msg;
  //     msg = (double *)malloc(sizeof(char)*blocklen);
  //   case default:
  //     printf("Error: mympi_Type not recognized.\n");
  //     exit(0);
  // }

  // int hndle = dsm_server_open(SOCK_STREAM, &port, FALSE, NULL, 0);
  while (len<blocklen) {
    // Receive data
    printf("handle:%d\n", handle);
    s = dsm_server_accept(handle, SOCK_STREAM, port, FALSE, nodes[tag], &len, &dummy, 0);
    len = dsm_read(s, SOCK_STREAM, msg, blocklen, NULL);
    printf("!!%d %d\n",blocklen, len);
    // Send acknowledgment
    // if (len==blocklen) {
    //   s1 = dsm_client_open(SOCK_STREAM, port + 1, FALSE, nodes[src], NULL);
    //   dsm_write(s1, SOCK_STREAM, ack, strlen(ack) + 1, NULL);
    // }
    // printf("%d\n", handle);
    // printf("%s\n", msg);
    // *status=(len>0)?1:0;
    // dsm_close(s);
    }

  // dsm_close(handle);

  // Get "blocklen" chuck of the message
}



// int main(int argc, char **argv) {
//   if (argc<3)
//     printf("Insufficient arguments. Usage %s 0(server)/1(client) TCP_PORT_NUMBER <msg>\n",argv[0]);
//
//   int s;
//   int dstport;
//   char *dsthost;
//   char **nodes=readNodes();
//   // char* msg = (char *)malloc(sizeof(char)*BUFSZ);
//   // s = dsm_client_open(SOCK_STREAM, SERV_PORT, FALSE, nodes[6], NULL);
//   if(atoi(argv[1])==0)
//     server(atoi(argv[2]));
//   else:
//   // printf("%s\n", msg);
//     if (argc<4)
//       printf("Message not recieved\n");
//       exit(0);
//     else
//       char* msg = argv[3];
//     client(msg, nodes[0], SERV_PORT);
// }
