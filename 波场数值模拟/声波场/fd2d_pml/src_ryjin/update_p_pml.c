`:@#include "fd.h"
#include "globalvar.h"

void update_p_pml()
{
    int ix, iz;
    float dvx_dx, dvz_dz;
    float dp_dx, dp_dz;
    float c2 = 0.0416666667;
    float c1 = 1.125;
    
    // 1. 更新 P (分裂场更新)
    for( ix=2; ix<nx-2; ix++ ) {
        for( iz=2; iz<nz-2; iz++ ) {
            // 计算导数（注意：这里不需要在导数里乘 dt，dt 统一在下面乘）
            dvx_dx = (c1*(aux_vx[ix*nz+iz] - aux_vx[(ix-1)*nz+iz]) - c2*(aux_vx[(ix+1)*nz+iz] - aux_vx[(ix-2)*nz+iz])) / dx;
            dvz_dz = (c1*(aux_vz[ix*nz+iz] - aux_vz[ix*nz+iz-1]) - c2*(aux_vz[ix*nz+iz+1] - aux_vz[ix*nz+iz-2])) / dz;

            // Px 只更新 x 分量，Pz 只更新 z 分量
            // 注意：这里 pml_vel 应该是 v*v，dt 在后面统一乘
            float v2 = pml_vel[ix*nz+iz]; // 假设你在 main 里没乘 dt*dt，如果乘了，这里要去掉
            
            aux_px[ix*nz+iz] = ( (1.0 - eta1_x[ix]*dt) * aux_px[ix*nz+iz] - v2 * dt * dvx_dx );
            aux_pz[ix*nz+iz] = ( (1.0 - eta1_z[iz]*dt) * aux_pz[ix*nz+iz] - v2 * dt * dvz_dz );
        }
    }

    // 2. 更新 V (速度场更新)
    for( ix=2; ix<nx-2; ix++ ) {
        for( iz=2; iz<nz-2; iz++ ) {
            // P = Px + Pz (更新 V 时使用总压力梯度)
            float p_total_curr = aux_px[ix*nz+iz] + aux_pz[ix*nz+iz];
            float p_total_right = aux_px[(ix+1)*nz+iz] + aux_pz[(ix+1)*nz+iz];
            float p_total_left = aux_px[(ix-1)*nz+iz] + aux_pz[(ix-1)*nz+iz];
            float p_total_far_right = aux_px[(ix+2)*nz+iz] + aux_pz[(ix+2)*nz+iz];

            dp_dx = (c1*(p_total_right - p_total_curr) - c2*(p_total_far_right - p_total_left)) / dx;
            
            // 同理更新 Vz...
            
            aux_vx[ix*nz+iz] = ( (1.0 - eta2_x[ix]*dt) * aux_vx[ix*nz+iz] - dt * dp_dx );
        }
    }
