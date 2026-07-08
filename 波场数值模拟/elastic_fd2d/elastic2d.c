#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX 401
#define NZ 401
#define NT 3401
#define DX 10.0f
#define DZ 10.0f
#define DT 0.0005f
#define IDX(ix, iz) ((ix) * NZ + (iz))
#define PI 3.14159265358979323846

static float *alloc_field(void) {
    float *p = (float *)calloc((size_t)NX * NZ, sizeof(float));
    if (!p) { fprintf(stderr, "allocation failed\n"); exit(1); }
    return p;
}

static float ricker(float t) {
    const float f0 = 20.0f, t0 = 0.05f;
    double a = PI * f0 * (t - t0);
    double a2 = a * a;
    return (float)((1.0 - 2.0 * a2) * exp(-a2));
}

static void write_bin(const char *path, const float *a, size_t n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite(a, sizeof(float), n, fp);
    fclose(fp);
}

static void build_model(const char *name, float *vp, float *vs, float *rho) {
    for (int ix = 0; ix < NX; ix++) {
        for (int iz = 0; iz < NZ; iz++) {
            float x = ix * DX, z = iz * DZ;
            float vpi = 2500.0f, vsi = 1200.0f, rhoi = 2200.0f;
            if (strcmp(name, "layer") == 0) {
                if (z >= 1500.0f) { vpi = 3500.0f; vsi = 1900.0f; rhoi = 2400.0f; }
                else { vpi = 2200.0f; vsi = 1100.0f; rhoi = 2100.0f; }
            } else if (strcmp(name, "fault") == 0) {
                float inter = 1500.0f;
                if (x > 1500.0f && x < 2800.0f) inter += 450.0f * (x - 1500.0f) / 1300.0f;
                else if (x >= 2800.0f) inter += 450.0f;
                if (z >= inter) { vpi = 3600.0f; vsi = 2000.0f; rhoi = 2450.0f; }
                else { vpi = 2200.0f; vsi = 1100.0f; rhoi = 2100.0f; }
            }
            vp[IDX(ix, iz)] = vpi; vs[IDX(ix, iz)] = vsi; rho[IDX(ix, iz)] = rhoi;
        }
    }
}

static void make_damp(float *damp) {
    const int nb = 35;
    for (int ix = 0; ix < NX; ix++) {
        for (int iz = 0; iz < NZ; iz++) {
            int d = ix;
            if (iz < d) d = iz;
            if (NX - 1 - ix < d) d = NX - 1 - ix;
            if (NZ - 1 - iz < d) d = NZ - 1 - iz;
            if (d < nb) {
                float x = (float)(nb - d) / nb;
                damp[IDX(ix, iz)] = expf(-0.018f * x * x);
            } else damp[IDX(ix, iz)] = 1.0f;
        }
    }
}

static void snap_name(char *buf, size_t n, const char *model, const char *comp, float t) {
    int k = (int)(t * 10.0f + 0.5f);
    snprintf(buf, n, "results/%s_%s_%dp%ds.bin", model, comp, k / 10, k % 10);
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "uniform";
    const int sx = 200;
    const int sz = strcmp(model, "uniform") == 0 ? 200 : 20;
    const int rz = strcmp(model, "uniform") == 0 ? 221 : 41;
    const float snap_times[5] = {0.4f, 0.8f, 1.2f, 1.4f, 1.6f};
    const size_t n = (size_t)NX * NZ;
    char path[256];

    float *vp = alloc_field(), *vs = alloc_field(), *rho = alloc_field();
    float *lam = alloc_field(), *mu = alloc_field(), *damp = alloc_field();
    float *vx = alloc_field(), *vz = alloc_field(), *sxx = alloc_field(), *szz = alloc_field(), *sxz = alloc_field();
    float *rec_vx = (float *)calloc((size_t)NX * NT, sizeof(float));
    float *rec_vz = (float *)calloc((size_t)NX * NT, sizeof(float));
    if (!rec_vx || !rec_vz) { fprintf(stderr, "record allocation failed\n"); exit(1); }

    build_model(model, vp, vs, rho);
    make_damp(damp);
    for (size_t i = 0; i < n; i++) {
        mu[i] = rho[i] * vs[i] * vs[i];
        lam[i] = rho[i] * (vp[i] * vp[i] - 2.0f * vs[i] * vs[i]);
    }
    snprintf(path, sizeof(path), "results/%s_vp.bin", model); write_bin(path, vp, n);
    snprintf(path, sizeof(path), "results/%s_vs.bin", model); write_bin(path, vs, n);

    int sn = 0;
    for (int it = 0; it < NT; it++) {
        for (int ix = 0; ix < NX - 1; ix++) for (int iz = 1; iz < NZ - 1; iz++) {
            int id = IDX(ix, iz);
            float dsxx_dx = (sxx[IDX(ix+1,iz)] - sxx[id]) / DX;
            float dsxz_dz = (sxz[id] - sxz[IDX(ix,iz-1)]) / DZ;
            vx[id] += DT * (dsxx_dx + dsxz_dz) / rho[id];
        }
        for (int ix = 1; ix < NX - 1; ix++) for (int iz = 0; iz < NZ - 1; iz++) {
            int id = IDX(ix, iz);
            float dsxz_dx = (sxz[id] - sxz[IDX(ix-1,iz)]) / DX;
            float dszz_dz = (szz[IDX(ix,iz+1)] - szz[id]) / DZ;
            vz[id] += DT * (dsxz_dx + dszz_dz) / rho[id];
        }
        for (int ix = 1; ix < NX - 1; ix++) for (int iz = 1; iz < NZ - 1; iz++) {
            int id = IDX(ix, iz);
            float dvx_dx = (vx[id] - vx[IDX(ix-1,iz)]) / DX;
            float dvz_dz = (vz[id] - vz[IDX(ix,iz-1)]) / DZ;
            sxx[id] += DT * ((lam[id] + 2.0f * mu[id]) * dvx_dx + lam[id] * dvz_dz);
            szz[id] += DT * (lam[id] * dvx_dx + (lam[id] + 2.0f * mu[id]) * dvz_dz);
        }
        for (int ix = 0; ix < NX - 1; ix++) for (int iz = 0; iz < NZ - 1; iz++) {
            int id = IDX(ix, iz);
            float dvx_dz = (vx[IDX(ix,iz+1)] - vx[id]) / DZ;
            float dvz_dx = (vz[IDX(ix+1,iz)] - vz[id]) / DX;
            sxz[id] += DT * mu[id] * (dvx_dz + dvz_dx);
        }
        float src = 2.0e7f * ricker(it * DT);
        sxx[IDX(sx, sz)] += src; szz[IDX(sx, sz)] += src;
        for (size_t i = 0; i < n; i++) { vx[i]*=damp[i]; vz[i]*=damp[i]; sxx[i]*=damp[i]; szz[i]*=damp[i]; sxz[i]*=damp[i]; }
        for (int ix = 0; ix < NX; ix++) { rec_vx[(size_t)ix * NT + it] = vx[IDX(ix, rz)]; rec_vz[(size_t)ix * NT + it] = vz[IDX(ix, rz)]; }
        if (sn < 5 && it == (int)(snap_times[sn] / DT + 0.5f)) {
            snap_name(path, sizeof(path), model, "vx", snap_times[sn]); write_bin(path, vx, n);
            snap_name(path, sizeof(path), model, "vz", snap_times[sn]); write_bin(path, vz, n);
            sn++;
        }
    }
    snprintf(path, sizeof(path), "results/%s_record_vx.bin", model); write_bin(path, rec_vx, (size_t)NX * NT);
    snprintf(path, sizeof(path), "results/%s_record_vz.bin", model); write_bin(path, rec_vz, (size_t)NX * NT);
    printf("finished %s\n", model);
    free(vp); free(vs); free(rho); free(lam); free(mu); free(damp); free(vx); free(vz); free(sxx); free(szz); free(sxz); free(rec_vx); free(rec_vz);
    return 0;
}
