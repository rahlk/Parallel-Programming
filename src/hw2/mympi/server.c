#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

#define MAXLINE 4096 /*max text line length*/
#define SERV_PORT 3000 /*port*/
#define LISTENQ 8 /*maximum number of client connections */

int main (int argc, char **argv)
{
 int i, listenfd, connfd, n;
 socklen_t clilen;
 char buf[MAXLINE];
 struct sockaddr_in cliaddr, servaddr;
 struct hostent *he;
 struct in_addr **addr_list;

 //creation of the socket
 listenfd = socket (AF_INET, SOCK_STREAM, 0);

 if ((he = gethostbyname(argv[1])) == NULL) {  // get the host info
     herror("gethostbyname");
     return 2;
 }

 // print information about this host:
 printf("Official name is: %s\n", he->h_name);
 printf("    IP addresses: ");
 addr_list = (struct in_addr **)he->h_addr_list;
 for(i = 0; addr_list[i] != NULL; i++) {
     printf("%s ", inet_ntoa(*addr_list[i]));
 }
 printf("\n");

 //preparation of the socket address
 servaddr.sin_family = AF_INET;
 servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
 // servaddr.sin_addr.s_addr = inet_addr(inet_ntoa(*addr_list[0]));
 servaddr.sin_port = htons(SERV_PORT);

 bind(listenfd, (struct sockaddr *) &servaddr, sizeof(servaddr));

 listen(listenfd, LISTENQ);

 printf("%s\n","Server running...waiting for connections.");

 for ( ; ; ) {

  clilen = sizeof(cliaddr);
  connfd = accept(listenfd, (struct sockaddr *) &cliaddr, &clilen);
  printf("%s\n","Received request...");

  while ( (n = recv(connfd, buf, MAXLINE,0)) > 0)  {
   printf("%s","String received from and resent to the client:");
   puts(buf);
   send(connfd, buf, n, 0);
  }

 if (n < 0) {
  perror("Read error");
  exit(1);
 }
 close(connfd);

 }
 //close listening socket
 close(listenfd);
}
