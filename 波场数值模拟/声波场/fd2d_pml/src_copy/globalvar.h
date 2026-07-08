/*
 * global variables in finite difference program
 */

    #ifndef GLOBALVAR_H
    #define GLOBALVAR_H

    extern int nx, nz;
    extern float dx, dz;
    extern int spunit; 
    extern int nt;
    extern float dt;
    extern float f0, ts;
    extern int shotbeg, shotend, shotintvl;
    extern int zs;
    extern int ksnp;
    extern int ksg;
    extern float tsnp;
    extern float vmax; 
    extern char velnm[100];
    extern char datnm[50];

    extern float * true_vel;
    extern float * padded_vel;
    extern float * pml_vel;
    extern float * ricker;

    extern float dtdx, dtdz;
    
    extern float * p;
    extern float * vx;
    extern float * vz;

    extern float * aux_px;
    extern float * aux_pz;
    extern float * aux_vx;
    extern float * aux_vz;
    extern float * eta1_x;
    extern float * eta2_x;
    extern float * eta1_z;
    extern float * eta2_z;

    extern float * sg;
    extern float * image;
    extern float * stack_img;

    extern float * psv_s;
    extern float * psv_n;
    extern float * psv_e;
    extern float * psv_w;
    
    #endif
