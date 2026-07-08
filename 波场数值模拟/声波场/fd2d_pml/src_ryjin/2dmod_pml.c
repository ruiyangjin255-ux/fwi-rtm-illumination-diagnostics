#include "fd.h"
#include "globalvar.h"

int main(void)
{
    int ix, iz;
    int it;
    int ishot;
    int xs;
    float ucvt;
    FILE * fp; 
    double dtime;
    char str[5];
    char snpnm[100];
    char datnm[100];

    // read parameters from parameter table 
    readpar();
    
    // pad the model to use PML or random boundary
    nx += 2*nrnd; 
    nz += 2*nrnd;
    zs += nrnd;

    // spacial unit converter, default is meter
    if ( spunit == 1 )
        ucvt = 1.0;
    else if ( spunit == 2 )
        ucvt = 1000.0;
    else
        ucvt = 0.3048;
    
    // allocate memory for vel, vx, vz and p
    true_vel = (float *) malloc((nx-2*nrnd)*(nz-2*nrnd)*sizeof(float));
    pml_vel = (float *) malloc(nx*nz*sizeof(float));
    if ( true_vel==NULL || pml_vel==NULL )
    {
        printf("Fail to allocate memory for vel, rnd_vel");
        exit(EXIT_FAILURE);
    }

    aux_px = (float *) malloc(nx*nz*sizeof(float));
    aux_pz = (float *) malloc(nx*nz*sizeof(float));
    aux_vx = (float *) malloc(nx*nz*sizeof(float));
    aux_vz = (float *) malloc(nx*nz*sizeof(float));
    if ( aux_px==NULL || aux_pz==NULL || aux_vx==NULL || aux_vz==NULL )
    {
        printf("Fail to allocate memory for vx, vz, p");
        exit(EXIT_FAILURE);
    }
   
    // allocate memory for source wavelet and surface data
    ricker = (float *) malloc(nt*sizeof(float));
    sg     = (float *) malloc((nx-2*nrnd)*nt*sizeof(float));
    if ( ricker==NULL || sg==NULL )
    {
        printf("Fail to allocate memory for ricker");
        exit(EXIT_FAILURE);
    }

    // read velocity of model from file  
    fp = fopen(velnm,"r");
    if( fp==NULL )
    {
        printf("Fail to open velocity file\n");
        exit(EXIT_FAILURE);
    }
    fread(true_vel,sizeof(float),(nx-2*nrnd)*(nz-2*nrnd),fp);
    fclose(fp);
            
    // convert unit all spacial variables to meter 
    if( spunit != 1 )
    {
        dx = dx*ucvt;
        dz = dz*ucvt;
        vmax = vmax*ucvt;
    }
    dtdx = dt/dx;
    dtdz = dt/dz;
    
    // find out the maximum velocity for stability
    vmax = 0.0;
    for ( ix=0; ix<nx-2*nrnd; ix++ )
        for ( iz=0; iz<nz-2*nrnd; iz++ )
            vmax = max(vmax,true_vel[ix*(nz-2*nrnd)+iz]);
    if ( vmax*dtdx >= 0.5 )
    {
        printf("Unstable, stop!\n");
        exit(EXIT_FAILURE);
    } 

    // pad the vel for using PML   
    pad(true_vel, pml_vel, 0);
// 在 main.c 中加入这行测试
printf("Debug: Velocity at center = %f\n", true_vel[((nx-2*nrnd)*(nz-2*nrnd))/2]);
    // proccess the rnd_vel to dt/dx*v*v 
    for ( ix=0; ix<nx; ix++ )
    {
        for ( iz=0; iz<nz; iz++ )
        {
            pml_vel[ix*nz+iz] = pml_vel[ix*nz+iz]*pml_vel[ix*nz+iz];
            pml_vel[ix*nz+iz] = dtdx*ucvt*ucvt*pml_vel[ix*nz+iz];
        }
    }
    
    // initialize PML
    eta1_x = (float *) malloc(nx*sizeof(float));
    eta2_x = (float *) malloc(nx*sizeof(float));
    eta1_z = (float *) malloc(nz*sizeof(float));
    eta2_z = (float *) malloc(nz*sizeof(float));
    if ( eta1_x==NULL || eta2_x==NULL || eta1_z==NULL || eta2_z==NULL )
    {
        printf("Fail to allocate memory for eta");
        exit(EXIT_FAILURE);
    }
    memset(eta1_x,0,nx*sizeof(float));
    memset(eta2_x,0,nx*sizeof(float));
    memset(eta1_z,0,nz*sizeof(float));
    memset(eta2_z,0,nz*sizeof(float));
    pml_boundary();

    // prepare source wavelet
    wavelet();
    
    dtime = -omp_get_wtime();
    // shot loop
    for( ishot=shotbeg; ishot<=shotend; ishot += shotintvl)
    {
        // initialize p, vx and vz 
        memset(aux_px,0,nx*nz*sizeof(float));
        memset(aux_pz,0,nx*nz*sizeof(float));
        memset(aux_vx,0,nx*nz*sizeof(float));
        memset(aux_vz,0,nx*nz*sizeof(float));
       
        xs = nrnd+ishot;

        // time loop for each shot
        for( it=0; it<nt; it++ )
        {
            if ( it%1000 == 0 ) 
                printf("Modeling, it = %d\n",it);
            
            // inject the source wavelet
            aux_px[xs*nz+zs] += ricker[it];
            aux_pz[xs*nz+zs] += ricker[it];
            
            // update p, vx and vz
            update_p_pml();

            // recorde surface data 
            for ( ix=nrnd; ix<nx-nrnd; ix++ )
                sg[(ix-nrnd)*nt+it] = aux_px[ix*nz+zs];

            // output the snapshot of a certain time
            if ( ksnp == 1 && it == (int) (tsnp/dt) )
            {
                sprintf(str, "%04d", ishot);
                strcpy(snpnm, "snapshot_");
                strcat(snpnm, str);
                strcat(snpnm, ".bin");
                fp = fopen(snpnm,"w");
                if( fp==NULL )
                {
                    printf("Fail to open forward_snapshot.bin\n");
                    exit(EXIT_FAILURE);
                }
                pad(true_vel,aux_px,1);
                fwrite(true_vel,sizeof(float),(nx-2*nrnd)*(nz-2*nrnd),fp);
                fclose(fp);
            }

        } // end time loop 
            
        // write data into disk 
        if ( ksg == 1 )
        {
            sprintf(str, "%04d", ishot);
            strcpy(datnm, "../data/data_");
            strcat(datnm, str);
            strcat(datnm, ".bin");
            fp = fopen(datnm,"w");
            if( fp==NULL )
            {
                printf("Fail to open file to write data\n");
                exit(EXIT_FAILURE);
            }
            fwrite(sg,sizeof(float),(nx-2*nrnd)*nt,fp);
            fclose(fp);
        }
    
    }
                
    dtime += omp_get_wtime(); 
    printf("Time: %f seconds\n",dtime); 

    free( true_vel ); 
    free( pml_vel ); 
    free( aux_px );
    free( aux_pz );
    free( aux_vx );
    free( aux_vz );
    free( ricker );
    free( eta1_x );
    free( eta2_x );
    free( eta1_z );
    free( eta2_z );
    free( sg );

    return 0;
}
