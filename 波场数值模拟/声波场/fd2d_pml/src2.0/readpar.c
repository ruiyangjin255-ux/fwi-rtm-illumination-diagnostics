#include "fd.h"
#include "globalvar.h"

void readpar(  )
{
    FILE * fp;
    char str[15];

    fp = fopen("../bin/partable.dat","r");
    if( fp==NULL )
    {
        printf("Fail to open partable.dat\n");
        exit(EXIT_FAILURE);
    }
    fscanf(fp,"%s = %d",str,&nx);
    fscanf(fp,"%s = %d",str,&nz);
    fscanf(fp,"%s = %f",str,&dx);
    fscanf(fp,"%s = %f",str,&dz);
    fscanf(fp,"%s = %d",str,&spunit);
    fscanf(fp,"%s = %f",str,&dt);
    fscanf(fp,"%s = %d",str,&nt);
    fscanf(fp,"%s = %f",str,&f0);
    fscanf(fp,"%s = %f",str,&ts);
    fscanf(fp,"%s = %d",str,&shotbeg);
    fscanf(fp,"%s = %d",str,&shotend);
    fscanf(fp,"%s = %d",str,&shotintvl);
    fscanf(fp,"%s = %d",str,&zs);
    fscanf(fp,"%s = %d",str,&ksnp);
    fscanf(fp,"%s = %f",str,&tsnp);
    fscanf(fp,"%s = %d",str,&ksg);
    fscanf(fp,"%s = %s",str,velnm);
    fscanf(fp,"%s = %s",str,datnm);
    fclose(fp);

    fp = fopen("./info.txt","w");
    if( fp==NULL )
    {
        printf("Fail to open info.txt\n");
        exit(EXIT_FAILURE);
    }
    fprintf(fp,"%s\n","******************info.txt*********************"); 
    fprintf(fp,"%s\n"," nx     nz     dx    dz ");
    fprintf(fp,"%2d %5d %6.3f %6.3f\n",nx, nz, dx, dz);
    fprintf(fp,"%s\n"," spunit ");
    fprintf(fp,"%2d\n",spunit);
    fprintf(fp,"%s\n"," dt     nt     f0    ts ");
    fprintf(fp,"%7.4f %5d %6.2f %6.2f\n",dt, nt, f0, ts);
    fprintf(fp,"%s\n"," shotbeg     shotend     shotintvl    zs ");
    fprintf(fp,"%3d %5d %5d %5d\n",shotbeg, shotend, shotintvl, zs);
    fprintf(fp,"%s\n"," ksnp     ksg     tsnp");
    fprintf(fp,"%2d %5d %6.3f\n",ksnp,  ksg, tsnp);
    fprintf(fp,"%s\n"," velocity file location ");
    fprintf(fp,"%s\n",velnm);
    fprintf(fp,"%s\n"," surface data file location ");
    fprintf(fp,"%s\n",datnm);
    fprintf(fp,"%s\n","***********************************************"); 
    fclose(fp); 

}

