/*
 * global variables in finite difference program
 */

    #include "globalvar.h"
    int nx, nz;
    float dx, dz;
    int spunit; 
    int nt;
    float dt;
    float f0, ts;
    int shotbeg, shotend, shotintvl;
    int zs;
    int ksnp;
    int ksg;
    float tsnp;
    float vmax; 
    char velnm[100];
    char datnm[50];

    float * true_vel;
    float * padded_vel;
    float * pml_vel;
    float * ricker;

    float dtdx, dtdz;
    
    float * p;
    float * vx;
    float * vz;

    float * aux_px;
    float * aux_pz;
    float * aux_vx;
    float * aux_vz;
    float * eta1_x;
    float * eta2_x;
    float * eta1_z;
    float * eta2_z;

    float * sg;
    float * image;
    float * stack_img;

    float * psv_s;
    float * psv_n;
    float * psv_e;
    float * psv_w;
