#include "fd.h"

// 定义所有在报错中出现的全局变量
int nx, nz, nt, spunit, zs, ishot, shotbeg, shotend, shotintvl, ksnp, ksg;
float dx, dz, dt, f0, ts, tsnp, vmax, dtdx, dtdz;
float *true_vel, *pml_vel, *aux_px, *aux_pz, *aux_vx, *aux_vz;
float *ricker, *eta1_x, *eta2_x, *eta1_z, *eta2_z;
float *sg;
char velnm[200], datnm[200];
