#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BUFSZ 4096 /*max text line length*/
#define SERV_PORT 3000 /*port*/

char** readNodes();

int main(int argc, char **argv) {
  char msg[BUFSZ];
  int s;
  int dstport;
  char *dsthost;
  char **nodes=readNodes();
  /* Sanity check*/
  // int j;
  printf("%s\n", nodes[0]);
  // for(j = 0; j < 1; j++)
    // printf("%s\n", nodes[j]);

}

char **readNodes(){
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

  FILE *fp = fopen("NODES", "r");
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
