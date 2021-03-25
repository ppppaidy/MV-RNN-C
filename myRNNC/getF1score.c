#include "getF1score.h"
#include <string.h>

double getF1score(
	int* predictLabel, int sentences,
	char** categories, int categories_len,
	int* sentenceLabels, int sentenceLabels_len,
	char* name, char* results
){
	char proposedFileStr[256];
	sprintf(proposedFileStr, "%sproposedAnswers%s.txt", results, name);
	FILE* proposedFile = fopen(proposedFileStr, "w");
	
	if(sentenceLabels_len == 0){
		printf("No labels, will just output predictions\n");
		for(int i = 0; i < sentences; i++)
			fprintf(proposedFile, "%d\t%s\n", i+1, categories[predictLabel[i]]);
		fclose(proposedFile);
		return -1;
	}
	
	char answerFileStr[256];
	sprintf(answerFileStr, "%sanswerKey%s.txt", results, name);
	FILE* answerFile = fopen(answerFileStr, "w");
	char outFileStr[256];
	sprintf(outFileStr, "%sresults%s.txt", results, name);
	for(int i = 0; i < sentences; i++){
		fprintf(proposedFile, "%d\t%s\n", i+1, categories[predictLabel[i]]);
		fprintf(answerFile, "%d\t%s\n", i+1, categories[sentenceLabels[i]-1]);
	}
	fclose(proposedFile);
	fclose(answerFile);
	
	char cmdString[1000];
	sprintf(cmdString, "./semeval2010_task8_scorer-v1.2.pl %s %s > %s", proposedFileStr, answerFileStr, outFileStr);
	system(cmdString);
	FILE* outFile = fopen(outFileStr, "r");
	char a[256] = {0}, b[256] = {0}, c[256] = {0};
	while(fscanf(outFile, "%s", a) == 1){
		strcpy(c, b);
		strcpy(b, a);
	}
	int c_len = strlen(c);
	c[c_len-1] = 0;
	double F1 = atof(c);
	fclose(outFile);
	
	return F1;
}
