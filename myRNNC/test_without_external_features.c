#include <stdio.h>
#include <stdlib.h>
#include "loaddata.h"
#include "forwardPropTree.h"
#include "getF1score.h"

const int maxWordLength = 50;

int main(int argc, char* argv[]){
	
	//load data
	double* Wv; int Wv_n, Wv_m;
	double* Wo; int Wo_n, Wo_m;
	double* W; int W_n, W_m;
	double* WO; int WO_n, WO_m;
	double* Wcat; int Wcat_n, Wcat_m;
	
	loaddata5(&Wv, &Wv_n, &Wv_m, "./tempdata/WvWOEF.txt");
	loaddata5(&Wo, &Wo_n, &Wo_m, "./tempdata/WoWOEF.txt");
	loaddata5(&W, &W_n, &W_m, "./tempdata/WWOEF.txt");
	loaddata5(&WO, &WO_n, &WO_m, "./tempdata/WOWOEF.txt");
	loaddata5(&Wcat, &Wcat_n, &Wcat_m, "./tempdata/WcatWOEF.txt");
	
	double params_regC, params_regC_Wcat, params_regC_WcatFeat;
	double params_regC_Wv, params_regC_Wo;
	int params_wordSize, params_rankWo, params_NN, params_numInContextWords;
	int params_numOutContextWords, params_tinyDataSet, params_categories;
	int params_test_without_external_features, params_test_with_external_features;
	char params_paths_data[256], params_paths_outputFolder[256];
	char params_paths_results[256], params_paths_dataFile[256];
	double params_features_mean[5], params_features_std[5];
	
	loadparams(
		&params_regC, &params_regC_Wcat, &params_regC_WcatFeat,
		&params_regC_Wv, &params_regC_Wo,
		&params_wordSize, &params_rankWo, &params_NN, &params_numInContextWords,
		&params_numOutContextWords, &params_tinyDataSet, &params_categories,
		&params_test_without_external_features, &params_test_with_external_features,
		params_paths_data, params_paths_outputFolder,
		params_paths_results, params_paths_dataFile,
		params_features_mean, params_features_std,
		"./tempdata/paramsWOEF.txt"
	);
	
	int** allSNum; int* allSNum_len; int allSNum_n;
	char*** allSStr; int* allSStr_len; int allSStr_n;
	int** allSTree; int* allSTree_len; int allSTree_n;
	int** allSNN; int* allSNN_len_n; int* allSNN_len_m; int allSNN_n;
	int* allIndicies; int allIndicies_n, allIndicies_m;
	char** categories; int categories_n, categories_m;
	int* sentenceLabels; int sentenceLabels_n, sentenceLabels_m;
	
	loaddata0(&allSNum, &allSNum_len, &allSNum_n, "./tempdata/allSNumWOEF.txt");
	loaddata1(&allSStr, &allSStr_len, &allSStr_n, maxWordLength, "./tempdata/allSStrWOEF.txt");
	loaddata0(&allSTree, &allSTree_len, &allSTree_n, "./tempdata/allSTreeWOEF.txt");
	loaddata2(&allSNN, &allSNN_len_n, &allSNN_len_m, &allSNN_n, "./tempdata/allSNNWOEF.txt");
	loaddata3(&allIndicies, &allIndicies_n, &allIndicies_m, "./tempdata/allIndiciesWOEF.txt");
	loaddata4(&categories, &categories_n, &categories_m, maxWordLength, "./tempdata/categoriesWOEF.txt");
	loaddata3(&sentenceLabels, &sentenceLabels_n, &sentenceLabels_m, "./tempdata/sentenceLabelsWOEF.txt");
	//load data end
	
	//test
	int sentences = allSNum_n;
	if(params_tinyDataSet) sentences = 10;
	int* predictLabel = (int*) malloc(sentences * sizeof(int));
	for(int i = 0; i < sentences; i++){
		forwardPropTree(
			allSNum[i], allSNum_len[i],
			allSTree[i], allSTree_len[i],
			allSStr[i], allSStr_len[i],
			allSNN[i], allSNN_len_n[i], allSNN_len_m[i],
			allIndicies + allIndicies_m * i, allIndicies_m,
			Wv, Wv_n, Wv_m,
			Wo, Wo_n, Wo_m,
			W, W_n, W_m,
			WO, WO_n, WO_m,
			Wcat, Wcat_n, Wcat_m,
			params_regC, params_regC_Wcat, params_regC_WcatFeat,
			params_regC_Wv, params_regC_Wo,
			params_wordSize, params_rankWo, params_NN, params_numInContextWords,
			params_numOutContextWords, params_tinyDataSet, params_categories,
			params_test_without_external_features, params_test_with_external_features,
			params_paths_data, params_paths_outputFolder,
			params_paths_results, params_paths_dataFile,
			params_features_mean, params_features_std,
			predictLabel + i,
			0
		);
	}
	
	double F1 = getF1score(predictLabel, sentences, categories, categories_m,
		sentenceLabels, sentenceLabels_n, "WOEF", params_paths_results);
	
	printf("F1 score without external features is: %.3lf\n", F1);
	
	//free
	free(Wv);
	free(Wo);
	free(W);
	free(WO);
	free(Wcat);
	intFree2(allSNum, allSNum_len, allSNum_n);
	stringFree2(allSStr, allSStr_len, allSStr_n);
	intFree2(allSTree, allSTree_len, allSTree_n);
	intFree3(allSNN, allSNN_len_n, allSNN_len_m, allSNN_n);
	free(allIndicies);
	stringFree1(categories, categories_n * categories_m);
	free(sentenceLabels);
	free(predictLabel);
	//free end
	
	return 0;
}
