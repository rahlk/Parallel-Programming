#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sock_msg.h"
#define BUFSZ 4096 /*max text line length*/
#define SERV_PORT 4096 /*port*/

// char** readNodes();

int client(char *msg, char *dsthost, int dstport) {
  // char msg[256];
  int s;
  // int dstport, char *dsthost;
  /* compose message, determine destination host/port */
  // memset(msg, 0, sizeof(msg));
  // printf("%s\n", msg);
  s = dsm_client_open(SOCK_STREAM, dstport, FALSE, dsthost, NULL);
  dsm_write(s, SOCK_STREAM, msg, strlen(msg) + 1, NULL);
  // printf("%d\n", s);
  dsm_close(s);
}

int server(int port) {
  int handle;//, port=SERV_PORT;
  int s, len;
  char msg[256], *dummy;
  handle = dsm_server_open(SOCK_STREAM, &port, FALSE, NULL, 0);

  while (1) {
    // printf("%d\n", s);
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

int main(int argc, char **argv) {
  if (argc<3)
    printf("Insufficient arguments. Usage %s 0(server)/1(client) TCP_PORT_NUMBER <msg>\n",argv[0]);

  int s;
  int dstport;
  char *dsthost;
  char **nodes = fileAsArray("NODES");
  char **ports = fileAsArray("PORTS");

  // char* msg = (char *)malloc(sizeof(char)*BUFSZ);
  // s = dsm_client_open(SOCK_STREAM, SERV_PORT, FALSE, nodes[6], NULL);
  if(atoi(argv[1])==0)
    server(atoi(argv[2]));

  else {
  // printf("%s\n", msg);
    if (argc<4) {
        printf("Message not recieved\n");
        exit(0);
    }
    else {
      char* msg = argv[3];
      client(msg, nodes[0], atoi(argv[2]));
    }
  }
}
