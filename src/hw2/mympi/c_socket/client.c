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

int main(int argc, char **argv)
{
 int sockfd;
 int i;
 struct sockaddr_in servaddr;
 struct hostent *he;
 struct in_addr **addr_list;
 char sendline[MAXLINE], recvline[MAXLINE];

 //basic check of the arguments
 //additional checks can be inserted
 if (argc !=2) {
  perror("Usage: TCPClient <IP address of the server");
  exit(1);
 }

 //Create a socket for the client
 //If sockfd<0 there was an error in the creation of the socket
 if ((sockfd = socket (AF_INET, SOCK_STREAM, 0)) <0) {
  perror("Problem in creating the socket");
  exit(2);
 }


 if ((he = gethostbyname(argv[1])) == NULL) {  // get the host info
     herror("gethostbyname");
     return 2;
 }

 // print information about this host:
 printf("Official name is: %s\n", he->h_name);
 printf("    IP addresses: ");
 addr_list = (struct in_addr **)he->h_addr_list;
 printf("%s ", inet_ntoa(*addr_list[0]));
 printf("\n");

 //Creation of the socket
 memset(&servaddr, 0, sizeof(servaddr));
 servaddr.sin_family = AF_INET;
 // servaddr.sin_addr.s_addr= inet_addr(argv[1]);
 // servaddr.sin_addr.s_addr= inet_addr(inet_ntoa(*addr_list[0]));
 servaddr.sin_addr.s_addr= inet_addr(inet_ntoa(*addr_list[0]));
 servaddr.sin_port =  htons(SERV_PORT); //convert to big-endian order
 printf("%d\n", sizeof(servaddr));
 //Connection of the client to the socket
 if (connect(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr))<0) {
  perror("Problem in connecting to the server");
  exit(3);
 }

 while (fgets(sendline, MAXLINE, stdin) != NULL) {

  send(sockfd, sendline, strlen(sendline), 0);

  if (recv(sockfd, recvline, MAXLINE,0) == 0){
   //error: server terminated prematurely
   perror("The server terminated prematurely");
   exit(4);
  }
  printf("%s", "String received from the server: ");
  fputs(recvline, stdout);
 }

 exit(0);
}
