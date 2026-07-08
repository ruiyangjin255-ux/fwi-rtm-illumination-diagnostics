#include "fd.h"
#include "globalvar.h"

void pad(float *v0, float *v, int key)
{
    int ix, iz;

    if ( key==0 )
    {
        for ( ix=nrnd; ix<nx-nrnd; ix++ )
            for ( iz=nrnd; iz<nz-nrnd; iz++ )
                v[ix*nz+iz] = v0[(ix-nrnd)*(nz-2*nrnd)+iz-nrnd];
    
        for ( ix=nrnd; ix<nx-nrnd; ix++ )
        {
            for( iz=0; iz<nrnd; iz++ )
                v[ix*nz+iz] = v[ix*nz+nrnd];
            for( iz=nz-nrnd; iz<nz; iz++ )
                v[ix*nz+iz] = v[ix*nz+nz-nrnd-1];
        }  

        for ( iz=0; iz<nz; iz++ )
        {
            for( ix=0; ix<nrnd; ix++ )
                v[ix*nz+iz] = v[nrnd*nz+iz];
            for( ix=nx-nrnd; ix<nx; ix++ )
                v[ix*nz+iz] = v[(nx-nrnd-1)*nz+iz];
        }
    }

    if( key==1 )
    {
        for ( ix=nrnd; ix<nx-nrnd; ix++ )
            for ( iz=nrnd; iz<nz-nrnd; iz++ )
                v0[(ix-nrnd)*(nz-2*nrnd)+iz-nrnd] = v[ix*nz+iz];

    }

}
