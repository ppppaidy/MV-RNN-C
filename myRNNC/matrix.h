#include <stdio.h>
#include <stdlib.h>

double* mat_plus(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m);

double* mat_mul(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m);

double* mat2vec(int* len, double* a, int a_n, int a_m, int start, int end, int col);

double* diag(int* n, int* m, double* a, int len);

double* reshape(int* n, int* m, double* a, int a_n, int a_m, int b_n, int b_m);

double* trans(int* n, int* m, double* a, int a_n, int a_m);

double* connect_r(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m);

double* connect_c(int* n, int* m, double* a, int a_n, int a_m, double* b, int b_n, int b_m);

double* eye(int* n, int * m, int r);

double* repmat(int* n, int* m, double* a, int a_n, int a_m, int x, int y);

double* coef(int* n, int* m, double* a, int a_n, int a_m, double co);

double* sub_mat(int* n, int* m, double* a, int a_n, int a_m, int s_x, int e_x, int s_y, int e_y);

double* mean_c(int* n, int* m, double* a, int a_n, int a_m);
