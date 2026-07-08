#include "fd.h"
#include "globalvar.h"

void update_p_pml()
{
    int ix, iz;
    float dvx_dx, dvz_dz;
    float dp_dx, dp_dz;
    float c2 = 0.0416666667;
    float c1 = 1.125;
    
    // update p
    #pragma omp parallel
    {
    #pragma omp for private(dvx_dx,dvz_dz)
    for( ix=2; ix<nx-2; ix++ )
    {
        for( iz=2; iz<nz-2; iz++ )
        {
            dvx_dx = c1*( aux_vx[ix*nz+iz] - aux_vx[(ix-1)*nz+iz] )
                   - c2*( aux_vx[(ix+1)*nz+iz] - aux_vx[(ix-2)*nz+iz] );
            dvz_dz = c1*( aux_vz[ix*nz+iz] - aux_vz[ix*nz+iz-1] )
                   - c2*( aux_vz[ix*nz+iz+1] - aux_vz[ix*nz+iz-2] );
            aux_px[ix*nz+iz] += -eta1_x[ix]*aux_px[ix*nz+iz]
                         + pml_vel[ix*nz+iz]*( dvx_dx + dvz_dz );
            aux_pz[ix*nz+iz] += -eta1_z[iz]*aux_pz[ix*nz+iz]
                         + pml_vel[ix*nz+iz]*( dvx_dx + dvz_dz );
        }
    }
   
    }

    // update vx vz 
    #pragma omp parallel
    {
    #pragma omp for private(dp_dx,dp_dz)
    for( ix=2; ix<nx-2; ix++ )
    {
        for( iz=2; iz<nz-2; iz++ )
        {
            dp_dx = c1*( aux_px[(ix+1)*nz+iz] - aux_px[ix*nz+iz] )
                   - c2*( aux_px[(ix+2)*nz+iz] - aux_px[(ix-1)*nz+iz] );
            dp_dz = c1*( aux_pz[ix*nz+iz+1] - aux_pz[ix*nz+iz] )
                   - c2*( aux_pz[ix*nz+iz+2] - aux_pz[ix*nz+iz-1] );
            aux_vx[ix*nz+iz] += -eta2_x[ix]*aux_vx[ix*nz+iz]+dtdx*dp_dx;
            aux_vz[ix*nz+iz] += -eta2_z[iz]*aux_vz[ix*nz+iz]+dtdx*dp_dz;
        }
    }
    
    }
}
