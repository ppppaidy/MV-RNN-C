#include "loaddata.h"

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
){
	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%lf", params_regC);
	fscanf(fp, "%lf", params_regC_Wcat);
	fscanf(fp, "%lf", params_regC_WcatFeat);
	fscanf(fp, "%lf", params_regC_Wv);
	fscanf(fp, "%lf", params_regC_Wo);
	fscanf(fp, "%d", params_wordSize);
	fscanf(fp, "%d", params_rankWo);
	fscanf(fp, "%d", params_NN);
	fscanf(fp, "%d", params_numInContextWords);
	fscanf(fp, "%d", params_numOutContextWords);
	fscanf(fp, "%d", params_tinyDataSet);
	fscanf(fp, "%d", params_categories);
	fscanf(fp, "%d", params_test_without_external_features);
	fscanf(fp, "%d", params_test_with_external_features);
	fscanf(fp, "%s", params_paths_data);
	fscanf(fp, "%s", params_paths_outputFolder);
	fscanf(fp, "%s", params_paths_results);
	fscanf(fp, "%s", params_paths_dataFile);
	for(int i = 0; i < 5; i++) fscanf(fp, "%lf", params_features_mean+i);
	for(int i = 0; i < 5; i++) fscanf(fp, "%lf", params_features_std+i);
	fclose(fp);
}

void loaddata0(int*** a, int** a_len, int* n, char* filename){
	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%d", n);
	(*a) = (int**) malloc((*n) * sizeof(int*));
	(*a_len) = (int*) malloc((*n) * sizeof(int));
	for(int i = 0; i < *n; i++){
		fscanf(fp, "%d", (*a_len) + i);
		(*a)[i] = (int*) malloc((*a_len)[i] * sizeof(int));
		for(int j = 0; j < (*a_len)[i]; j++)
			fscanf(fp, "%d", (*a)[i] + j);
	}
	fclose(fp);
}

void loaddata1(char**** a, int** a_len, int* n, int wordsize, char* filename){
	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%d", n);
	(*a) = (char***) malloc((*n) * sizeof(char**));
	(*a_len) = (int*) malloc((*n) * sizeof(int));
	for(int i = 0; i < *n; i++){
		fscanf(fp, "%d", (*a_len) + i);
		(*a)[i] = (char**) malloc((*a_len)[i] * sizeof(char*));
		for(int j = 0; j < (*a_len)[i]; j++){
			(*a)[i][j] = (char*) malloc(wordsize * sizeof(char));
			fscanf(fp, "%s", (*a)[i][j]);
		}
	}
	fclose(fp);
}

void loaddata2(int*** a, int** a_len_n, int** a_len_m, int* n, char* filename){
	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%d", n);
	(*a) = (int**) malloc((*n) * sizeof(int*));
	(*a_len_n) = (int*) malloc((*n) * sizeof(int));
	(*a_len_m) = (int*) malloc((*n) * sizeof(int));
	for(int i = 0; i < *n; i++){
		fscanf(fp, "%d %d", (*a_len_n) + i, (*a_len_m) + i);
		(*a)[i] = (int*) malloc((*a_len_n)[i] * (*a_len_m)[i] * sizeof(int));
		for(int j = 0; j < (*a_len_n)[i]*(*a_len_m)[i]; j++)
			fscanf(fp, "%d", (*a)[i] + j);
	}
	fclose(fp);
}

void loaddata3(int** a, int* n, int* m, char* filename){
	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%d%d", n, m);
	(*a) = (int*) malloc((*n) * (*m) * sizeof(int));
	for(int i = 0; i < (*n) * (*m); i++) fscanf(fp, "%d", (*a) + i);
	fclose(fp);
}

void loaddata4(char*** a, int* n, int* m, int wordsize, char* filename){
	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%d%d", n, m);
	(*a) = (char**) malloc((*n) * (*m) * sizeof(char*));
	for(int i = 0; i < (*n) * (*m); i++){
		(*a)[i] = malloc(wordsize * sizeof(char));
		fscanf(fp, "%s", (*a)[i]);
	}
	fclose(fp);
}

void loaddata5(double** a, int* n, int* m, char* filename){
	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%d%d", n, m);
	(*a) = (double*) malloc((*n) * (*m) * sizeof(double));
	for(int i = 0; i < (*n) * (*m); i++) fscanf(fp, "%lf", (*a) + i);
	fclose(fp);
}

void stringFree1(char** a, int len){
	for(int i = 0; i < len; i++) free(a[i]);
	free(a);
}

void stringFree2(char*** a, int* a_len, int len){
	for(int i = 0; i < len; i++){
		for(int j = 0; j < a_len[i]; j++) free(a[i][j]);
		free(a[i]);
	}
	free(a);
	free(a_len);
}

void intFree2(int** a, int* a_len, int len){
	for(int i = 0; i < len; i++) free(a[i]);
	free(a);
	free(a_len);
}

void intFree3(int** a, int* a_len_n, int* a_len_m, int len){
	for(int i = 0; i < len; i++) free(a[i]);
	free(a);
	free(a_len_n);
	free(a_len_m);
}

