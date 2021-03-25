#include "findPath.h"
#include "matrix.h"

int check(int* path1, int path1_len, int parent){
	for(int i = 0; i < path1_len; i++)
		if(path1[i] == parent) return 1;
	return 0;
}

int* findPath(int* len, int* pp, int* indicies){
	int elem1 = indicies[0];
	int elem2 = indicies[1];
	
	int path1_len = 0;
	int parent = pp[elem1];
	while(parent != -1){
		path1_len++;
		parent = pp[parent];
	}
	path1_len++;
	
	int* path1 = (int*) malloc(path1_len * sizeof(int));
	path1_len = 0;
	parent = pp[elem1];
	while(parent != -1){
		path1[path1_len++] = parent;
		parent = pp[parent];
	}
	path1[path1_len++] = -1;
	
	int path2_len = 0;
	parent = pp[elem2];
	while(!check(path1, path1_len, parent)){
		path2_len++;
		parent = pp[parent];
	}
	
	int fullPath_len = 0;
	for(int i = 0; i < path1_len; i++){
		fullPath_len++;
		if(path1[i] == parent) break;
	}
	
	int* fullPath = (int*) malloc((fullPath_len + path2_len) * sizeof(int));
	
	for(int i = 0; i < fullPath_len; i++)
		fullPath[i] = path1[i];
	fullPath_len += path2_len;
	parent = elem2;
	for(int i = 0; i < path2_len; i++){
		parent = pp[parent];
		fullPath[fullPath_len-i-1] = parent;
	}
	
	*len = fullPath_len;
	
	free(path1);
	return fullPath;
}
