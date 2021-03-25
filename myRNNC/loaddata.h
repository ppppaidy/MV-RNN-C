#include <stdio.h>
#include <stdlib.h>

void loadparams(
	double* params_regC, double* params_regC_Wcat, 
	double* params_regC_WcatFeat, double* params_regC_Wv, double* params_regC_Wo,
	int* params_wordSize, int* params_rankWo, int* params_NN, int* params_numInContextWords,
	int* params_numOutContextWords, int* params_tinyDataSet, int* params_categories,
	int* params_test_without_external_features, int* params_test_with_external_features,
	char* params_paths_data, char* params_paths_outputFolder,
	char* params_paths_results, char* params_paths_dataFile,
	double* params_features_mean, double* params_features_std,
	char* filename
);

void loaddata0(int*** a, int** a_len, int* n, char* filename);

void loaddata1(char**** a, int** a_len, int* n, int wordsize, char* filename);

void loaddata2(int*** a, int** a_len_n, int** a_len_m, int* n, char* filename);

void loaddata3(int** a, int* n, int* m, char* filename);

void loaddata4(char*** a, int* n, int* m, int wordsize, char* filename);

void loaddata5(double** a, int* n, int* m, char* filename);

void stringFree1(char** a, int len);

void stringFree2(char*** a, int* a_len, int len);

void intFree2(int** a, int* a_len, int len);

void intFree3(int** a, int* a_len_n, int* a_len_m, int len);
