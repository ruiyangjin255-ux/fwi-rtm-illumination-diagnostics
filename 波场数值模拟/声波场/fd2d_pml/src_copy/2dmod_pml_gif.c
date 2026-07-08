#include "fd.h"
#include "globalvar.h"

static void write_snapshot(int ishot, int it)
{
    FILE *fp;
    char snpnm[128];

    snprintf(snpnm, sizeof(snpnm), "gif_snapshot_%04d_%04d.bin", ishot, it);
    fp = fopen(snpnm, "wb");
    if (fp == NULL)
    {
        printf("Fail to open %s\n", snpnm);
        exit(EXIT_FAILURE);
    }
    pad(true_vel, aux_px, 1);
    fwrite(true_vel, sizeof(float), (nx - 2 * nrnd) * (nz - 2 * nrnd), fp);
    fclose(fp);
}

int main(void)
{
    int ix, iz;
    int it;
    int ishot;
    int xs;
    int snap_step;
    float ucvt;
    FILE *fp;
    double dtime;

    readpar();

    nx += 2 * nrnd;
    nz += 2 * nrnd;
    zs += nrnd;

    if (spunit == 1)
        ucvt = 1.0;
    else if (spunit == 2)
        ucvt = 1000.0;
    else
        ucvt = 0.3048;

    true_vel = (float *)malloc((nx - 2 * nrnd) * (nz - 2 * nrnd) * sizeof(float));
    pml_vel = (float *)malloc(nx * nz * sizeof(float));
    aux_px = (float *)malloc(nx * nz * sizeof(float));
    aux_pz = (float *)malloc(nx * nz * sizeof(float));
    aux_vx = (float *)malloc(nx * nz * sizeof(float));
    aux_vz = (float *)malloc(nx * nz * sizeof(float));
    ricker = (float *)malloc(nt * sizeof(float));
    sg = (float *)malloc((nx - 2 * nrnd) * nt * sizeof(float));
    if (true_vel == NULL || pml_vel == NULL || aux_px == NULL || aux_pz == NULL ||
        aux_vx == NULL || aux_vz == NULL || ricker == NULL || sg == NULL)
    {
        printf("Fail to allocate wavefield arrays\n");
        exit(EXIT_FAILURE);
    }

    fp = fopen(velnm, "rb");
    if (fp == NULL)
    {
        printf("Fail to open velocity file: %s\n", velnm);
        exit(EXIT_FAILURE);
    }
    fread(true_vel, sizeof(float), (nx - 2 * nrnd) * (nz - 2 * nrnd), fp);
    fclose(fp);

    if (spunit != 1)
    {
        dx = dx * ucvt;
        dz = dz * ucvt;
        vmax = vmax * ucvt;
    }
    dtdx = dt / dx;
    dtdz = dt / dz;

    vmax = 0.0;
    for (ix = 0; ix < nx - 2 * nrnd; ix++)
        for (iz = 0; iz < nz - 2 * nrnd; iz++)
            vmax = max(vmax, true_vel[ix * (nz - 2 * nrnd) + iz]);
    if (vmax * dtdx >= 0.5)
    {
        printf("Unstable, stop!\n");
        exit(EXIT_FAILURE);
    }

    pad(true_vel, pml_vel, 0);
    for (ix = 0; ix < nx; ix++)
    {
        for (iz = 0; iz < nz; iz++)
        {
            pml_vel[ix * nz + iz] = pml_vel[ix * nz + iz] * pml_vel[ix * nz + iz];
            pml_vel[ix * nz + iz] = dtdx * ucvt * ucvt * pml_vel[ix * nz + iz];
        }
    }

    eta1_x = (float *)malloc(nx * sizeof(float));
    eta2_x = (float *)malloc(nx * sizeof(float));
    eta1_z = (float *)malloc(nz * sizeof(float));
    eta2_z = (float *)malloc(nz * sizeof(float));
    if (eta1_x == NULL || eta2_x == NULL || eta1_z == NULL || eta2_z == NULL)
    {
        printf("Fail to allocate PML coefficients\n");
        exit(EXIT_FAILURE);
    }
    memset(eta1_x, 0, nx * sizeof(float));
    memset(eta2_x, 0, nx * sizeof(float));
    memset(eta1_z, 0, nz * sizeof(float));
    memset(eta2_z, 0, nz * sizeof(float));
    pml_boundary();
    wavelet();

    snap_step = (int)(tsnp / dt + 0.5);
    if (snap_step < 1)
        snap_step = 1;

    dtime = -omp_get_wtime();
    for (ishot = shotbeg; ishot <= shotend; ishot += shotintvl)
    {
        memset(aux_px, 0, nx * nz * sizeof(float));
        memset(aux_pz, 0, nx * nz * sizeof(float));
        memset(aux_vx, 0, nx * nz * sizeof(float));
        memset(aux_vz, 0, nx * nz * sizeof(float));
        memset(sg, 0, (nx - 2 * nrnd) * nt * sizeof(float));

        xs = nrnd + ishot;
        for (it = 0; it < nt; it++)
        {
            if (it % 1000 == 0)
                printf("Modeling, it = %d\n", it);

            aux_px[xs * nz + zs] += ricker[it];
            aux_pz[xs * nz + zs] += ricker[it];
            update_p_pml();

            if (ksnp == 1 && it > 0 && it % snap_step == 0)
                write_snapshot(ishot, it);
        }
    }

    dtime += omp_get_wtime();
    printf("Time: %f seconds\n", dtime);

    free(true_vel);
    free(pml_vel);
    free(aux_px);
    free(aux_pz);
    free(aux_vx);
    free(aux_vz);
    free(ricker);
    free(eta1_x);
    free(eta2_x);
    free(eta1_z);
    free(eta2_z);
    free(sg);

    return 0;
}
