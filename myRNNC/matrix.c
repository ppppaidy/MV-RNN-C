#include "matrix.h"

double* mat_plus(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m){
	*n = a_n;
	*m = a_m;
	double* c = (double*) malloc((*n) * (*m) * sizeof(double));
	for(int i = 0; i < (*n) * (*m); i++) c[i] = a[i] + b[i];
	return c;
}

double* mat_mul(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m){
	*n = a_n;
	*m = b_m;
	double* c = (double*) calloc((*n) * (*m), sizeof(double));
	for(int i = 0; i < (*n); i++)
		for(int j = 0; j < (*m); j++)
			for(int k = 0; k < a_m; k++)
				c[i*(*m)+j] += a[i*a_m+k] * b[k*b_m+j];
	return c;
}

double* mat2vec(int* len, double* a, int a_n, int a_m, int start, int end, int col){
	*len = end - start;
	double* c = (double*) malloc((*len) * sizeof(double));
	for(int i = 0; i < (*len); i++)
		c[i] = a[(i+start)*a_m + col];
	return c;
}

double* diag(int* n, int* m, double* a, int len){
	*n = len;
	*m = len;
	double* c = (double*) calloc((*n) * (*m), sizeof(double));
	for(int i = 0; i < len; i++) c[i*(*m)+i] = a[i];
	return c;
}

double* reshape(int* n, int* m, double* a, int a_n, int a_m, int b_n, int b_m){
	*n = b_n;
	*m = b_m;
	double* c = (double*) malloc((*n) * (*m) * sizeof(double));
	int x = -1; int y = 0;
	for(int j = 0; j < (*m); j++)
		for(int i = 0; i < (*n); i++){
			x++;
			if(x == a_n) {x = 0; y++;}
			c[i*(*m)+j] = a[x*a_m+y];
		}
	return c;
}

double* trans(int* n, int* m, double* a, int a_n, int a_m){
	*n = a_m;
	*m = a_n;
	double* c = (double*) malloc((*n) * (*m) * sizeof(double));
	for(int i = 0; i < (*n); i++)
		for(int j = 0; j < (*m); j++)
			c[i*(*m)+j] = a[j*a_m+i];
	return c;
}

double* connect_r(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m){
	*n = a_n + b_n;
	*m = a_m;
	double* c = (double*) malloc((*n) * (*m) * sizeof(double));
	for(int i = 0; i < a_n; i++)
		for(int j = 0; j < a_m; j++)
			c[i*(*m)+j] = a[i*a_m+j];
	for(int i = 0; i < b_n; i++)
		for(int j = 0; j < b_m; j++)
			c[(i+a_n)*(*m)+j] = b[i*b_m+j];
	return c;
}

double* connect_c(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m){
	*n = a_n;
	*m = a_m + b_m;
	double* c = (double*) malloc((*n) * (*m) * sizeof(double));
	for(int i = 0; i < a_n; i++)
		for(int j = 0; j < a_m; j++)
			c[i*(*m)+j] = a[i*a_m+j];
	for(int i = 0; i < b_n; i++)
		for(int j = 0; j < b_m; j++)
			c[i*(*m)+j+a_m] = b[i*b_m+j];
	return c;
}

double* eye(int* n, int * m, int r){
	*n = r;
	*m = r;
	double* c = (double*) calloc((*n) * (*m), sizeof(double));
	for(int i = 0; i < r; i++) c[i*r+i] = 1;
	return c;
}

double* repmat(int* n, int* m, double* a, int a_n, int a_m, int x, int y){
	*n = a_n * x;
	*m = a_m * y;
	double* c= (double*) malloc((*n) * (*m) * sizeof(double));
	for(int l = 0; l < x; l++)
		for(int r = 0; r < y; r++)
			for(int i = 0; i < a_n; i++)
				for(int j = 0; j < a_m; j++)
					c[(l*a_m+i)*(*m)+r*a_n+j] = a[i*a_m+j];
	return c;
}

double* coef(int* n, int* m, double* a, int a_n, int a_m, double co){
	*n = a_n;
	*m = a_m;
	double* c= (double*) malloc((*n) * (*m) * sizeof(double));
	for(int i = 0; i < (*n); i++)
		for(int j = 0; j < (*m); j++)
			c[i*(*m)+j] = a[i*a_m+j] * co;
	return c;
}

double* sub_mat(int* n, int* m, double* a, int a_n, int a_m, int s_x, int e_x, int s_y, int e_y){
	*n = e_x - s_x;
	*m = e_y - s_y;
	double* c = (double*) malloc((*n) * (*m) * sizeof(double));
	for(int i = 0; i < (*n); i++)
		for(int j = 0; j < (*m); j++)
			c[i*(*m)+j] = a[(i+s_x)*a_m+j+s_y];
	return c;
}

double* mean_c(int* n, int* m, double* a, int a_n, int a_m){
	*n = a_n;
	*m = 1;
	double* c = (double*) calloc((*n) * (*m), sizeof(double));
	for(int i = 0; i < a_n; i++)
		for(int j = 0; j < a_m; j++)
			c[i] += a[i*a_m+j];
	for(int i = 0; i < a_n; i++)
		c[i] /= a_m;
	return c;
}

