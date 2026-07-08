#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define max(x,y) (x>y)?x:y 

#define PI 3.141592653589793
#define nrnd 50
#define npml 25
#define pmlq 1.0
#define NumBound 3

void readpar(void);
void wavelet(void);
void update_p_pml(void);
void backupdate_p(float *, float *, float *);
void pad(float *, float *, int);
void pml_boundary(void);
