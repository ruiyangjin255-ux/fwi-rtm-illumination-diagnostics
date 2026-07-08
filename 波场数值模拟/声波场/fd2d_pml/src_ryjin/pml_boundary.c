#include "fd.h"
#include "globalvar.h"

void pml_boundary()
{
    int ix, iz;
    float tmp;
    
    // npml decressing coefs in x direction
    for ( ix=0; ix<npml; ix++ )
    {
        tmp = (npml-1-ix)/(float)npml;
        eta1_x[ix] = pmlq*tmp*tmp*tmp;
        tmp = (npml-1-ix-0.5)/(float)npml;
        eta2_x[ix] = pmlq*tmp*tmp*tmp;
    }
    
    for ( ix=nx-npml; ix<nx; ix++ )
    {
        tmp = (ix-nx+npml)/(float)npml;
        eta1_x[ix] = pmlq*tmp*tmp*tmp;
        tmp = (ix-nx+npml+0.5)/(float)npml;
        eta2_x[ix] = pmlq*tmp*tmp*tmp;
    }
    
    // npml decressing coefs in z direction
    for ( iz=0; iz<npml; iz++ )
    {
        tmp = (npml-1-iz)/(float)npml;
        eta1_z[iz] = pmlq*tmp*tmp*tmp;
        tmp = (npml-1-iz-0.5)/(float)npml;
        eta2_z[iz] = pmlq*tmp*tmp*tmp;
    }
    
    for ( iz=nz-npml; iz<nz; iz++ )
    {
        tmp = (iz-nz+npml)/(float)npml;
        eta1_z[iz] = pmlq*tmp*tmp*tmp;
        tmp = (iz-nz+npml+0.5)/(float)npml;
        eta2_z[iz] = pmlq*tmp*tmp*tmp;
    }
}
