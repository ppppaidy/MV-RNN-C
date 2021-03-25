#include "getInternalFeatures.h"

void getInternalFeatures(
	double* features, int* nodePath, int nodePath_len,
	int sStr_len, int* indicies, double* features_mean, double* features_std
){
	int depth1 = 0;
	for(int i = 0; i < nodePath_len; i++)
		if(nodePath[i] > nodePath[depth1]) depth1 = i;
	depth1 += 2;
	int depth2 = nodePath_len - depth1 + 3;
	features[0] = nodePath_len + 2;
	features[1] = depth1;
	features[2] = depth2;
	features[3] = sStr_len;
	features[4] = indicies[1] - indicies[0];
	for(int i = 0; i < 5; i++)
		features[i] = (features[i] - features_mean[i]) / features_std[i];
}
