#include <stdio.h>
#include <stdlib.h>

double getF1score(
	int* predictLabel, int sentences,
	char** categories, int categories_len,
	int* sentenceLabels, int sentenceLabels_len,
	char* name, char* results
);
