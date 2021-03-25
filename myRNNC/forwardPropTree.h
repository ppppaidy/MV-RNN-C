#include <stdio.h>
#include <stdlib.h>

void forwardPropTree(
	int* sNum, int sNum_len,
	int* sTree, int sTree_len,
	char** sStr, int sStr_len,
	int* sNN, int sNN_len_n, int sNN_len_m,
	int* indicies, int indicies_len,
	double* Wv, int Wv_n, int Wv_m,
	double* Wo, int Wo_n, int Wo_m,
	double* W, int W_n, int W_m,
	double* WO, int WO_n, int WO_m,
	double* Wcat, int Wcat_n, int Wcat_m,
	double params_regC, double params_regC_Wcat, 
	double params_regC_WcatFeat, double params_regC_Wv, double params_regC_Wo,
	int params_wordSize, int params_rankWo, int params_NN, int params_numInContextWords,
	int params_numOutContextWords, int params_tinyDataSet, int params_categories,
	int params_test_without_external_features, int params_test_with_external_features,
	char* params_paths_data, char* params_paths_outputFolder,
	char* params_paths_results, char* params_paths_dataFile,
	double* params_features_mean, double* params_features_std,
	int* predictLabel,
	int onlyGetVectors
);
