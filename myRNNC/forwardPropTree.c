#include "forwardPropTree.h"
#include "matrix.h"
#include "findPath.h"
#include "getInternalFeatures.h"
#include <math.h>

int max(int a, int b){
	return a>b? a:b;
}

int min(int a, int b){
	return a<b? a:b;
}

void addOuterContext(
	double** thisTree_poolMatrix, int* thisTree_poolMatrix_n, int* thisTree_poolMatrix_m,
	int** thisTree_nodePath, int* thisTree_nodePath_len,
	double** thisTree_pooledVecPath, int* thisTree_pooledVecPath_n, int* thisTree_pooledVecPath_m,
	double* thisTree_nodeAct_a, int thisTree_nodeAct_a_n, int thisTree_nodeAct_a_m,
	int* indicies, int sentlen, int n, int wsz
);

void addInnerContext(
	double** thisTree_poolMatrix, int* thisTree_poolMatrix_n, int* thisTree_poolMatrix_m,
	int** thisTree_nodePath, int* thisTree_nodePath_len,
	double** thisTree_pooledVecPath, int* thisTree_pooledVecPath_n, int* thisTree_pooledVecPath_m,
	double* thisTree_nodeAct_a, int thisTree_nodeAct_a_n, int thisTree_nodeAct_a_m,
	double* Wv, int Wv_n, int Wv_m,
	int* indicies, int sentlen, int n, int wsz
);

void addNN(
	double** thisTree_poolMatrix, int* thisTree_poolMatrix_n, int* thisTree_poolMatrix_m,
	double** thisTree_NN_vecs, int* thisTree_NN_vecs_n, int *thisTree_NN_vecs_m,
	double* Wv, int Wv_n, int Wv_m,
	int* sNN, int sNN_len_n, int sNN_len_m,
	int wsz, int NN
);

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
){
	int wsz = params_wordSize;
	int r = params_rankWo;
	
	for(int i = 0; i < sTree_len; i++) sTree[i]--;
	for(int i = 0; i < indicies_len; i++) indicies[i]--;
	
	int words_len = 0;
	for(int i = 0; i < sNum_len; i++)
		if(sNum[i] > 0) words_len++;
	int* words = (int*) malloc(words_len * sizeof(int));
	words_len = 0;
	for(int i = 0; i < sNum_len; i++)
		if(sNum[i] > 0) words[words_len++] = i;
	int numTotalNodes = sNum_len;
	
	double* allV = (double*) malloc(Wv_n * words_len * sizeof(double));
	int allV_n = Wv_n, allV_m = words_len;
	for(int i = 0; i < allV_n; i++)
		for(int j = 0; j < allV_m; j++)
			allV[i*allV_m + j] = Wv[i*Wv_m + sNum[words[j]]-1];
	double* allO = (double*) malloc(Wo_n * words_len * sizeof(double));
	int allO_n = Wo_n, allO_m = words_len;
	for(int i = 0; i < allO_n; i++)
		for(int j = 0; j < allO_m; j++)
			allO[i*allO_m + j] = Wo[i*Wo_m + sNum[words[j]]-1];
	
	int* thisTree_isLeafVec = (int*) calloc(numTotalNodes, sizeof(int));
	int thisTree_isLeafVec_len = numTotalNodes;
	for(int i = 0; i < words_len; i++)
		thisTree_isLeafVec[words[i]] = 1;
	
	int thisTree_nodeNames = sTree_len;
	int* thisTree_nodeLabels = (int*) malloc(sNum_len * sizeof(int));
	int thisTree_nodeLabels_len = sNum_len;
	for(int i = 0; i < sNum_len; i++)
		thisTree_nodeLabels[i] = sNum[i]-1;
	
	int* thisTree_ParIn_z = (int*) calloc(wsz * numTotalNodes, sizeof(int));
	int thisTree_ParIn_z_n = wsz, thisTree_ParIn_z_m = numTotalNodes;
	
	int* thisTree_ParIn_a = (int*) calloc(wsz * numTotalNodes, sizeof(int));
	int thisTree_ParIn_a_n = wsz, thisTree_ParIn_a_m = numTotalNodes;
	
	double* thisTree_nodeOp_A = (double*) calloc(wsz * wsz * numTotalNodes, sizeof(double));
	int thisTree_nodeOp_A_n = wsz*wsz, thisTree_nodeOp_A_m = numTotalNodes;
	
	int* thisTree_score = (int*) calloc(numTotalNodes, sizeof(int));
	int thisTree_score_n = numTotalNodes;
	
	int* thisTree_kids = (int*) calloc(numTotalNodes*2, sizeof(int));
	int thisTree_kids_n = numTotalNodes, thisTree_kids_m = 2;
	
	double* thisTree_nodeAct_a = (double*) malloc(allV_n * numTotalNodes * sizeof(double));
	int thisTree_nodeAct_a_n = allV_n, thisTree_nodeAct_a_m = numTotalNodes;
	for(int i = 0; i < allV_n; i++)
		for(int j = 0; j < allV_m; j++)
			thisTree_nodeAct_a[i*numTotalNodes+j] = allV[i*allV_m+j];
	
	for(int thisWordNum = 0; thisWordNum < words_len; thisWordNum++){
		int tmp1_len;
		double* tmp1 = mat2vec(&tmp1_len, allO, allO_n, allO_m, 0, wsz, thisWordNum);
		int tmp2_n, tmp2_m;
		double* tmp2 = diag(&tmp2_n, &tmp2_m, tmp1, tmp1_len);
		int tmp3_len;
		double* tmp3 = mat2vec(&tmp3_len, allO, allO_n, allO_m, wsz, wsz*(1+r), thisWordNum);
		int tmp4_n, tmp4_m;
		double* tmp4 = reshape(&tmp4_n, &tmp4_m, tmp3, tmp3_len, 1, wsz, r);
		int tmp5_len;
		double* tmp5 = mat2vec(&tmp5_len, allO, allO_n, allO_m, wsz*(1+r), allO_n, thisWordNum);
		int tmp6_n, tmp6_m;
		double* tmp6 = reshape(&tmp6_n, &tmp6_m, tmp5, tmp5_len, 1, wsz, r);
		int tmp7_n, tmp7_m;
		double* tmp7 = trans(&tmp7_n, &tmp7_m, tmp6, tmp6_n, tmp6_m);
		int tmp8_n, tmp8_m;
		double* tmp8 = mat_mul(&tmp8_n, &tmp8_m, tmp4, tmp4_n, tmp4_m, tmp7, tmp7_n, tmp7_m);
		int tmp9_n, tmp9_m;
		double* tmp9 = mat_plus(&tmp9_n, &tmp9_m, tmp2, tmp2_n, tmp2_m, tmp8, tmp8_n, tmp8_m);
		int tmp10_n, tmp10_m;
		double* tmp10 = reshape(&tmp10_n, &tmp10_m, tmp9, tmp9_n, tmp9_m, wsz*wsz, 1);
		for(int i = 0; i < wsz*wsz; i++)
			thisTree_nodeOp_A[i*thisTree_nodeOp_A_m+thisWordNum] = tmp10[i];
		free(tmp1);free(tmp2);free(tmp3);free(tmp4);free(tmp5);
		free(tmp6);free(tmp7);free(tmp8);free(tmp9);free(tmp10);
	}
	
	int* toMerge = words;
	int toMerge_len = words_len;
	
	while(toMerge_len > 1){
		int i = -1;
		int foundGoodPair = 0;
		while(!foundGoodPair){
			i++;
			if(sTree[toMerge[i]] == sTree[toMerge[i+1]])
				foundGoodPair = 1;
		}
		int newParent = sTree[toMerge[i+1]];
		int kid1 = toMerge[i];
		int kid2 = toMerge[i+1];
		thisTree_kids[newParent*thisTree_kids_m] = kid1;
		thisTree_kids[newParent*thisTree_kids_m+1] = kid2;
		toMerge[i] = newParent;
		for(int k = i+1; k < toMerge_len-1; k++)
			toMerge[k] = toMerge[k+1];
		toMerge_len--;
		
		int a_n, a_m = 1;
		double* a = mat2vec(&a_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n, 
			thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, kid1);
		int tmp1_n, tmp1_m = 1;
		double* tmp1 = mat2vec(&tmp1_n, thisTree_nodeOp_A, thisTree_nodeOp_A_n,
			thisTree_nodeOp_A_m, 0, thisTree_nodeOp_A_n, kid1);
		int A_n, A_m;
		double* A = reshape(&A_n, &A_m, tmp1, tmp1_n, tmp1_m, wsz, wsz);
		
		int b_n, b_m = 1;
		double* b = mat2vec(&b_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n, 
			thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, kid2);
		int tmp2_n, tmp2_m = 1;
		double* tmp2 = mat2vec(&tmp2_n, thisTree_nodeOp_A, thisTree_nodeOp_A_n,
			thisTree_nodeOp_A_m, 0, thisTree_nodeOp_A_n, kid2);
		int B_n, B_m;
		double* B = reshape(&B_n, &B_m, tmp2, tmp2_n, tmp2_m, wsz, wsz);
		
		int l_a_n, l_a_m;
		double* l_a = mat_mul(&l_a_n, &l_a_m, B, B_n, B_m, a, a_n, a_m);
		int r_a_n, r_a_m;
		double* r_a = mat_mul(&r_a_n, &r_a_m, A, A_n, A_m, b, b_n, b_m);
		
		int tmp3_n, tmp3_m;
		double* tmp3 = eye(&tmp3_n, &tmp3_m, 1);
		int tmp4_n, tmp4_m;
		double* tmp4 = connect_r(&tmp4_n, &tmp4_m, r_a, r_a_n, r_a_m, tmp3, tmp3_n, tmp3_m);
		int tmp5_n, tmp5_m;
		double* tmp5 = connect_r(&tmp5_n, &tmp5_m, l_a, l_a_n, l_a_m, tmp4, tmp4_n, tmp4_m);
		int tmp6_n, tmp6_m;
		double* tmp6 = mat_mul(&tmp6_n, &tmp6_m, W, W_n, W_m, tmp5, tmp5_n, tmp5_m);
		for(int i = 0; i < tmp6_n * tmp6_m; i++)
			tmp6[i] = tanh(tmp6[i]);
		for(int i = 0; i < tmp6_n; i++)
			thisTree_nodeAct_a[i*thisTree_nodeAct_a_m+newParent] = tmp6[i];
		
		int tmp7_n, tmp7_m;
		double* tmp7 = connect_r(&tmp7_n, &tmp7_m, A, A_n, A_m, B, B_n, B_m);
		int tmp8_n, tmp8_m;
		double* tmp8 = mat_mul(&tmp8_n, &tmp8_m, WO, WO_n, WO_m, tmp7, tmp7_n, tmp7_m);
		int P_A_n, P_A_m;
		double* P_A = reshape(&P_A_n, &P_A_m, tmp8, tmp8_n, tmp8_m, wsz*wsz, 1);
		
		for(int i = 0; i < l_a_n; i++)
			thisTree_ParIn_a[i*thisTree_ParIn_a_m+kid1] = l_a[i];
		for(int i = 0; i < r_a_n; i++)
			thisTree_ParIn_a[i*thisTree_ParIn_a_m+kid2] = r_a[i];
		for(int i = 0; i < P_A_n; i++)
			thisTree_nodeOp_A[i*thisTree_nodeOp_A_m+newParent] = P_A[i];
		
		free(tmp1);free(tmp2);free(a);free(b);
		free(A);free(B);free(l_a);free(r_a);
		free(tmp3);free(tmp4);free(tmp5);free(tmp6);
		free(tmp7);free(tmp8);free(P_A);
	}
	
	int thisTree_nodePath_len;
	int* thisTree_nodePath = findPath(&thisTree_nodePath_len, sTree, indicies);
	
	double thisTree_features[5];
	getInternalFeatures(thisTree_features, thisTree_nodePath, thisTree_nodePath_len,
		sStr_len, indicies, params_features_mean, params_features_std);
	
	int thisTree_nodePath_M = 0;
	for(int i = 0; i < thisTree_nodePath_len; i++)
		if(thisTree_nodePath[i] > thisTree_nodePath_M)
			thisTree_nodePath_M = thisTree_nodePath[i];
	
	int tmp1_n, tmp1_m = 1;
	double* tmp1 = mat2vec(&tmp1_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
		thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, thisTree_nodePath_M);
	int tmp2_n, tmp2_m = 1;
	double* tmp2 = mat2vec(&tmp2_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
		thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, indicies[0]);
	int tmp3_n, tmp3_m = 1;
	double* tmp3 = mat2vec(&tmp3_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
		thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, indicies[1]);
	int tmp4_n, tmp4_m;
	double* tmp4 = connect_c(&tmp4_n, &tmp4_m, tmp1, tmp1_n, tmp1_m, tmp2, tmp2_n, tmp2_m);
	int thisTree_pooledVecPath_n, thisTree_pooledVecPath_m;
	double* thisTree_pooledVecPath = connect_c(&thisTree_pooledVecPath_n,
		&thisTree_pooledVecPath_m, tmp4, tmp4_n, tmp4_m, tmp3, tmp3_n, tmp3_m);
	
	free(thisTree_nodePath);
	thisTree_nodePath_len = 3;
	thisTree_nodePath = (int*) malloc(3 * sizeof(int));
	thisTree_nodePath[0] = thisTree_nodePath_M;
	thisTree_nodePath[1] = indicies[0];
	thisTree_nodePath[2] = indicies[1];
	int thisTree_poolMatrix_n, thisTree_poolMatrix_m;
	double* thisTree_poolMatrix = eye(&thisTree_poolMatrix_n, &thisTree_poolMatrix_m, 3*wsz);
	
	addOuterContext(&thisTree_poolMatrix, &thisTree_poolMatrix_n, &thisTree_poolMatrix_m,
		&thisTree_nodePath, &thisTree_nodePath_len,
		&thisTree_pooledVecPath, &thisTree_pooledVecPath_n, &thisTree_pooledVecPath_m,
		thisTree_nodeAct_a, thisTree_nodeAct_a_n, thisTree_nodeAct_a_m,
		indicies, sStr_len, params_numOutContextWords, params_wordSize
	);
	
	addInnerContext(&thisTree_poolMatrix, &thisTree_poolMatrix_n, &thisTree_poolMatrix_m,
		&thisTree_nodePath, &thisTree_nodePath_len,
		&thisTree_pooledVecPath, &thisTree_pooledVecPath_n, &thisTree_pooledVecPath_m,
		thisTree_nodeAct_a, thisTree_nodeAct_a_n, thisTree_nodeAct_a_m,
		Wv, Wv_n, Wv_m,
		indicies, sStr_len, params_numInContextWords, params_wordSize
	);
	
	int thisTree_NN_vecs_n, thisTree_NN_vecs_m;
	double* thisTree_NN_vecs;
	
	addNN(&thisTree_poolMatrix, &thisTree_poolMatrix_n, &thisTree_poolMatrix_m,
		&thisTree_NN_vecs, &thisTree_NN_vecs_n, &thisTree_NN_vecs_m,
		Wv, Wv_n, Wv_m,
		sNN, sNN_len_n, sNN_len_m,
		params_wordSize, params_NN
	);
	
	int tmp5_n, tmp5_m;
	double* tmp5 = reshape(&tmp5_n, &tmp5_m, thisTree_pooledVecPath, thisTree_pooledVecPath_n,
		thisTree_pooledVecPath_m, thisTree_pooledVecPath_n*thisTree_pooledVecPath_m, 1);
	int tmp6_n, tmp6_m;
	double* tmp6 = reshape(&tmp6_n, &tmp6_m, thisTree_NN_vecs, thisTree_NN_vecs_n,
		thisTree_NN_vecs_m, thisTree_NN_vecs_n * thisTree_NN_vecs_m, 1);
	int tmp7_n, tmp7_m;
	double* tmp7 = connect_r(&tmp7_n, &tmp7_m, tmp6, tmp6_n, tmp6_m, thisTree_features, 5, 1);
	int catInput_n, catInput_m;
	double* catInput = connect_r(&catInput_n, &catInput_m, tmp5, tmp5_n, tmp5_m, tmp7, tmp7_n, tmp7_m);
	if(!onlyGetVectors){
		int num_n, num_m;
		double* num = mat_mul(&num_n, &num_m, Wcat, Wcat_n, Wcat_m, catInput, catInput_n, catInput_m);
		for(int i = 0; i < num_n; i++) num[i] = exp(num[i]);
		double numsum = 0;
		for(int i = 0; i < num_n; i++) numsum += num[i];
		for(int i = 0; i < num_n; i++) num[i] /= numsum;
		if(predictLabel != NULL){
			int t = 0;
			for(int i = 0; i < num_n; i++)
				if(num[i] > num[t]) t = i;
			*predictLabel = t;
		}
		free(num);
	}
	
	//free
	free(words);
	free(allV);
	free(allO);
	free(thisTree_isLeafVec);
	free(thisTree_nodeLabels);
	free(thisTree_ParIn_z);
	free(thisTree_ParIn_a);
	free(thisTree_nodeOp_A);
	free(thisTree_score);
	free(thisTree_kids);
	free(thisTree_nodeAct_a);
	free(thisTree_nodePath);
	free(tmp1);free(tmp2);free(tmp3);free(tmp4);
	free(thisTree_pooledVecPath);
	free(thisTree_poolMatrix);
	free(thisTree_NN_vecs);
	free(tmp5);free(tmp6);free(tmp7);
	free(catInput);
	//free end
}

void addOuterContext(
	double** thisTree_poolMatrix, int* thisTree_poolMatrix_n, int* thisTree_poolMatrix_m,
	int** thisTree_nodePath, int* thisTree_nodePath_len,
	double** thisTree_pooledVecPath, int* thisTree_pooledVecPath_n, int* thisTree_pooledVecPath_m,
	double* thisTree_nodeAct_a, int thisTree_nodeAct_a_n, int thisTree_nodeAct_a_m,
	int* indicies, int sentlen, int n, int wsz
){
	int wszI_n, wszI_m;
	double* wszI = eye(&wszI_n, &wszI_m, wsz);
	
	int o1L = max(indicies[0] - max(indicies[0] - n, 0), 1);
	int o1s = max(indicies[0] - n, 0), o1e = o1s + o1L;
	int tmp1_n = *thisTree_poolMatrix_n, tmp1_m = wsz;
	double* tmp1 = (double*) calloc(tmp1_n * tmp1_m, sizeof(double));
	int tmp2_n = o1L * wsz, tmp2_m = *thisTree_poolMatrix_m;
	double* tmp2 = (double*) calloc(tmp2_n * tmp2_m, sizeof(double));
	int tmp3_n, tmp3_m;
	double* tmp3 = coef(&tmp3_n, &tmp3_m, wszI, wszI_n, wszI_m, (double)1/o1L);
	int tmp4_n, tmp4_m;
	double* tmp4 = repmat(&tmp4_n, &tmp4_m, tmp3, tmp3_n, tmp3_m, o1L, 1);
	int tmp5_n, tmp5_m;
	double* tmp5 = connect_c(&tmp5_n, &tmp5_m, *thisTree_poolMatrix, *thisTree_poolMatrix_n,
		*thisTree_poolMatrix_m, tmp1, tmp1_n, tmp1_m);
	int tmp6_n, tmp6_m;
	double* tmp6 = connect_c(&tmp6_n, &tmp6_m, tmp2, tmp2_n, tmp2_m, tmp4, tmp4_n, tmp4_m);
	free(*thisTree_poolMatrix);
	if(indicies[0] == max(indicies[0] - n, 0)){
		*thisTree_poolMatrix = tmp5;
		*thisTree_poolMatrix_n = tmp5_n;
		*thisTree_poolMatrix_m = tmp5_m;
	}
	else{
		*thisTree_poolMatrix = connect_r(thisTree_poolMatrix_n, thisTree_poolMatrix_m,
			tmp5, tmp5_n, tmp5_m, tmp6, tmp6_n, tmp6_m);
		free(tmp5);
	}
	free(tmp1);free(tmp2);free(tmp3);free(tmp4);free(tmp6);
	
	int o2L = min(indicies[1]+n, sentlen-1) - indicies[1];
	int o2s = indicies[1] + 1, o2e = o2s + o2L;
	tmp1_n = *thisTree_poolMatrix_n, tmp1_m = wsz;
	tmp1 = (double*) calloc(tmp1_n * tmp1_m, sizeof(double));
	tmp2_n = o2L * wsz, tmp2_m = *thisTree_poolMatrix_m;
	tmp2 = (double*) calloc(tmp2_n * tmp2_m, sizeof(double));
	tmp3 = coef(&tmp3_n, &tmp3_m, wszI, wszI_n, wszI_m, (double)1/o2L);
	tmp4 = repmat(&tmp4_n, &tmp4_m, tmp3, tmp3_n, tmp3_m, o2L, 1);
	tmp5 = connect_c(&tmp5_n, &tmp5_m, *thisTree_poolMatrix, *thisTree_poolMatrix_n,
		*thisTree_poolMatrix_m, tmp1, tmp1_n, tmp1_m);
	tmp6 = connect_c(&tmp6_n, &tmp6_m, tmp2, tmp2_n, tmp2_m, tmp4, tmp4_n, tmp4_m);
	free(*thisTree_poolMatrix);
	*thisTree_poolMatrix = connect_r(thisTree_poolMatrix_n, thisTree_poolMatrix_m,
		tmp5, tmp5_n, tmp5_m, tmp6, tmp6_n, tmp6_m);
	free(tmp1);free(tmp2);free(tmp3);free(tmp4);free(tmp5);free(tmp6);
	
	int tmp7_len = *thisTree_nodePath_len + o1L + o2L;
	int* tmp7 = (int*) malloc(tmp7_len * sizeof(int));
	for(int i = 0; i < *thisTree_nodePath_len; i++)
		tmp7[i] = (*thisTree_nodePath)[i];
	for(int i = 0; i < o1L; i++)
		tmp7[*thisTree_nodePath_len+i] = o1s + i;
	for(int i = 0; i < o2L; i++)
		tmp7[*thisTree_nodePath_len+o1L+i] = o2s + i;
	free(*thisTree_nodePath);
	*thisTree_nodePath = tmp7;
	*thisTree_nodePath_len = tmp7_len;
	
	int tmp8_n, tmp8_m;
	double* tmp8 = sub_mat(&tmp8_n, &tmp8_m, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
		thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, o1s, o1e);
	int tmp9_n, tmp9_m;
	double* tmp9 = mean_c(&tmp9_n, &tmp9_m, tmp8, tmp8_n, tmp8_m);
	int tmp10_n, tmp10_m;
	double* tmp10 = sub_mat(&tmp10_n, &tmp10_m, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
		thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, o2s, o2e);
	int tmp11_n, tmp11_m;
	double* tmp11 = mean_c(&tmp11_n, &tmp11_m, tmp10, tmp10_n, tmp10_m);
	int tmp12_n, tmp12_m;
	double* tmp12 = connect_c(&tmp12_n, &tmp12_m, tmp9, tmp9_n, tmp9_m, tmp11, tmp11_n, tmp11_m);
	int tmp13_n, tmp13_m;
	double* tmp13 = connect_c(&tmp13_n, &tmp13_m, *thisTree_pooledVecPath, *thisTree_pooledVecPath_n,
		*thisTree_pooledVecPath_m, tmp12, tmp12_n, tmp12_m);
	free(*thisTree_pooledVecPath);
	*thisTree_pooledVecPath = tmp13;
	*thisTree_pooledVecPath_n = tmp13_n;
	*thisTree_pooledVecPath_m = tmp13_m;
	free(tmp8);free(tmp9);free(tmp10);free(tmp11);free(tmp12);
	free(wszI);
}

void addInnerContext(
	double** thisTree_poolMatrix, int* thisTree_poolMatrix_n, int* thisTree_poolMatrix_m,
	int** thisTree_nodePath, int* thisTree_nodePath_len,
	double** thisTree_pooledVecPath, int* thisTree_pooledVecPath_n, int* thisTree_pooledVecPath_m,
	double* thisTree_nodeAct_a, int thisTree_nodeAct_a_n, int thisTree_nodeAct_a_m,
	double* Wv, int Wv_n, int Wv_m,
	int* indicies, int sentlen, int n, int wsz
){
	int o1L = min(indicies[0]+n, sentlen-1) - indicies[0];
	int o1s = indicies[0] + 1;
	int vecs1_n, vecs1_m;
	double* vecs1;
	if(o1L < n){
		int tmp1_len = *thisTree_nodePath_len + n;
		int* tmp1 = (int*) malloc(tmp1_len * sizeof(int));
		for(int i = 0; i < *thisTree_nodePath_len; i++)
			tmp1[i] = (*thisTree_nodePath)[i];
		for(int i = 0; i < o1L; i++)
			tmp1[i+*thisTree_nodePath_len] = o1s + i;
		for(int i = o1L; i < n; i++)
			tmp1[i+*thisTree_nodePath_len] = sentlen-1;
		free(*thisTree_nodePath);
		*thisTree_nodePath = tmp1;
		*thisTree_nodePath_len = tmp1_len;
		if(o1L > 0){
			int tmp2_n, tmp2_m;
			double* tmp2 = sub_mat(&tmp2_n, &tmp2_m, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
				thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, o1s, o1s+o1L);
			int tmp3_n, tmp3_m = 1;
			double* tmp3 = mat2vec(&tmp3_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
				thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, sentlen-1);
			int tmp4_n, tmp4_m;
			double* tmp4 = repmat(&tmp4_n, &tmp4_m, tmp3, tmp3_n, tmp3_m, 1, n-o1L);
			vecs1 = connect_c(&vecs1_n, &vecs1_m, tmp2, tmp2_n, tmp2_m, tmp4, tmp4_n, tmp4_m);
			free(tmp2);free(tmp3);free(tmp4);
		}
		else{
			int tmp3_n, tmp3_m = 1;
			double* tmp3 = mat2vec(&tmp3_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
				thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, sentlen-1);
			vecs1 = repmat(&vecs1_n, &vecs1_m, tmp3, tmp3_n, tmp3_m, 1, n);
			free(tmp3);
		}
	}
	else{
		int tmp1_len = *thisTree_nodePath_len + o1L;
		int* tmp1 = (int*) malloc(tmp1_len * sizeof(int));
		for(int i = 0; i < *thisTree_nodePath_len; i++)
			tmp1[i] = (*thisTree_nodePath)[i];
		for(int i = 0; i < o1L; i++)
			tmp1[i+*thisTree_nodePath_len] = o1s + i;
		free(*thisTree_nodePath);
		*thisTree_nodePath = tmp1;
		*thisTree_nodePath_len = tmp1_len;
		vecs1 = sub_mat(&vecs1_n, &vecs1_m, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
			thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, o1s, o1s+o1L);
	}
	
	int tmp3_n = *thisTree_poolMatrix_n, tmp3_m = wsz*n;
	double* tmp3 = (double*) calloc(tmp3_n * tmp3_m, sizeof(double));
	int tmp4_n = wsz*n, tmp4_m = *thisTree_poolMatrix_m;
	double* tmp4 = (double*) calloc(tmp4_n * tmp4_m, sizeof(double));
	int tmp5_n, tmp5_m;
	double* tmp5 = eye(&tmp5_n, &tmp5_m, wsz*n);
	int tmp6_n, tmp6_m;
	double* tmp6 = connect_c(&tmp6_n, &tmp6_m, *thisTree_poolMatrix, *thisTree_poolMatrix_n,
		*thisTree_poolMatrix_m, tmp3, tmp3_n, tmp3_m);
	int tmp7_n, tmp7_m;
	double* tmp7 = connect_c(&tmp7_n, &tmp7_m, tmp4, tmp4_n, tmp4_m, tmp5, tmp5_n, tmp5_m);
	int tmp8_n, tmp8_m;
	double* tmp8 = connect_r(&tmp8_n, &tmp8_m, tmp6, tmp6_n, tmp6_m, tmp7, tmp7_n, tmp7_m);
	free(*thisTree_poolMatrix);
	*thisTree_poolMatrix = tmp8;
	*thisTree_poolMatrix_n = tmp8_n;
	*thisTree_poolMatrix_m = tmp8_m;
	free(tmp3);free(tmp4);free(tmp6);free(tmp7);
	
	int o2L = indicies[1] - max(indicies[1] - n, 0);
	int o2s = max(indicies[1] - n, 0);
	int vecs2_n, vecs2_m;
	double* vecs2;
	if(o2L < n){
		int tmp1_len = *thisTree_nodePath_len + n;
		int* tmp1 = (int*) malloc(tmp1_len * sizeof(int));
		for(int i = 0; i < *thisTree_nodePath_len; i++)
			tmp1[i] = (*thisTree_nodePath)[i];
		for(int i = o2L; i < n; i++)
			tmp1[i+*thisTree_nodePath_len-o2L] = -1;
		for(int i = 0; i < o2L; i++)
			tmp1[i+*thisTree_nodePath_len+n-o2L] = o2s + i;
		free(*thisTree_nodePath);
		*thisTree_nodePath = tmp1;
		*thisTree_nodePath_len = tmp1_len;
		if(o2L > 0){
			int tmp2_n, tmp2_m;
			double* tmp2 = sub_mat(&tmp2_n, &tmp2_m, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
				thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, o2s, o2s+o2L);
			int tmp3_n, tmp3_m = 1;
			double* tmp3 = mat2vec(&tmp3_n, Wv, Wv_n, Wv_m, 0, Wv_n, 0);
			int tmp4_n, tmp4_m;
			double* tmp4 = repmat(&tmp4_n, &tmp4_m, tmp3, tmp3_n, tmp3_m, 1, n-o2L);
			vecs2 = connect_c(&vecs2_n, &vecs2_m, tmp4, tmp4_n, tmp4_m, tmp2, tmp2_n, tmp2_m);
			free(tmp2);free(tmp3);free(tmp4);
		}
		else{
			int tmp3_n, tmp3_m = 1;
			double* tmp3 = mat2vec(&tmp3_n, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
				thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, 0);
			vecs2 = repmat(&vecs2_n, &vecs2_m, tmp3, tmp3_n, tmp3_m, 1, n);
			free(tmp3);
		}
	}
	else{
		int tmp1_len = *thisTree_nodePath_len + o2L;
		int* tmp1 = (int*) malloc(tmp1_len * sizeof(int));
		for(int i = 0; i < *thisTree_nodePath_len; i++)
			tmp1[i] = (*thisTree_nodePath)[i];
		for(int i = 0; i < o2L; i++)
			tmp1[i+*thisTree_nodePath_len] = o2s + i;
		free(*thisTree_nodePath);
		*thisTree_nodePath = tmp1;
		*thisTree_nodePath_len = tmp1_len;
		vecs2 = sub_mat(&vecs2_n, &vecs2_m, thisTree_nodeAct_a, thisTree_nodeAct_a_n,
			thisTree_nodeAct_a_m, 0, thisTree_nodeAct_a_n, o2s, o2s+o2L);
	}
	
	tmp3_n = *thisTree_poolMatrix_n, tmp3_m = wsz*n;
	tmp3 = (double*) calloc(tmp3_n * tmp3_m, sizeof(double));
	tmp4_n = wsz*n, tmp4_m = *thisTree_poolMatrix_m;
	tmp4 = (double*) calloc(tmp4_n * tmp4_m, sizeof(double));
	tmp6_n, tmp6_m;
	tmp6 = connect_c(&tmp6_n, &tmp6_m, *thisTree_poolMatrix, *thisTree_poolMatrix_n,
		*thisTree_poolMatrix_m, tmp3, tmp3_n, tmp3_m);
	tmp7_n, tmp7_m;
	tmp7 = connect_c(&tmp7_n, &tmp7_m, tmp4, tmp4_n, tmp4_m, tmp5, tmp5_n, tmp5_m);
	tmp8_n, tmp8_m;
	tmp8 = connect_r(&tmp8_n, &tmp8_m, tmp6, tmp6_n, tmp6_m, tmp7, tmp7_n, tmp7_m);
	free(*thisTree_poolMatrix);
	*thisTree_poolMatrix = tmp8;
	*thisTree_poolMatrix_n = tmp8_n;
	*thisTree_poolMatrix_m = tmp8_m;
	free(tmp3);free(tmp4);free(tmp5);free(tmp6);free(tmp7);
	
	int tmp1_n, tmp1_m;
	double* tmp1 = connect_c(&tmp1_n, &tmp1_m, *thisTree_pooledVecPath, *thisTree_pooledVecPath_n,
		*thisTree_pooledVecPath_m, vecs1, vecs1_n, vecs1_m);
	int tmp2_n, tmp2_m;
	double* tmp2 = connect_c(&tmp2_n, &tmp2_m, tmp1, tmp1_n, tmp1_m, vecs2, vecs2_n, vecs2_m);
	free(*thisTree_pooledVecPath);
	*thisTree_pooledVecPath = tmp2;
	*thisTree_pooledVecPath_n = tmp2_n;
	*thisTree_pooledVecPath_m = tmp2_m;
	free(tmp1);free(vecs1);free(vecs2);
}

void addNN(
	double** thisTree_poolMatrix, int* thisTree_poolMatrix_n, int* thisTree_poolMatrix_m,
	double** thisTree_NN_vecs, int* thisTree_NN_vecs_n, int *thisTree_NN_vecs_m,
	double* Wv, int Wv_n, int Wv_m,
	int* sNN, int sNN_len_n, int sNN_len_m,
	int wsz, int NN
){
	*thisTree_NN_vecs_n = Wv_n;
	*thisTree_NN_vecs_m = 2;
	*thisTree_NN_vecs = (double*) calloc(Wv_n * 2, sizeof(double));
	for(int i = 0; i < Wv_n; i++){
		for(int j = 0; j < NN; j++)
			(*thisTree_NN_vecs)[i*2] += Wv[i*Wv_m+sNN[j*2]-1];
		(*thisTree_NN_vecs)[i*2] /= NN;
	}
	for(int i = 0; i < Wv_n; i++){
		for(int j = 0; j < NN; j++)
			(*thisTree_NN_vecs)[i*2+1] += Wv[i*Wv_m+sNN[j*2+1]-1];
		(*thisTree_NN_vecs)[i*2+1] /= NN;
	}
	
	int tmp3_n = *thisTree_poolMatrix_n, tmp3_m = wsz*2;
	double* tmp3 = (double*) calloc(tmp3_n * tmp3_m, sizeof(double));
	int tmp4_n = wsz*2, tmp4_m = *thisTree_poolMatrix_m;
	double* tmp4 = (double*) calloc(tmp4_n * tmp4_m, sizeof(double));
	int tmp5_n, tmp5_m;
	double* tmp5 = eye(&tmp5_n, &tmp5_m, wsz*2);
	for(int i = 0; i < tmp5_n*tmp5_m; i++) tmp5[i] /= NN;
	int tmp6_n, tmp6_m;
	double* tmp6 = connect_c(&tmp6_n, &tmp6_m, *thisTree_poolMatrix, *thisTree_poolMatrix_n,
		*thisTree_poolMatrix_m, tmp3, tmp3_n, tmp3_m);
	int tmp7_n, tmp7_m;
	double* tmp7 = connect_c(&tmp7_n, &tmp7_m, tmp4, tmp4_n, tmp4_m, tmp5, tmp5_n, tmp5_m);
	int tmp8_n, tmp8_m;
	double* tmp8 = connect_r(&tmp8_n, &tmp8_m, tmp6, tmp6_n, tmp6_m, tmp7, tmp7_n, tmp7_m);
	free(*thisTree_poolMatrix);
	*thisTree_poolMatrix = tmp8;
	*thisTree_poolMatrix_n = tmp8_n;
	*thisTree_poolMatrix_m = tmp8_m;
	free(tmp3);free(tmp4);free(tmp5);free(tmp6);free(tmp7);
}

