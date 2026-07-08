#include "fd.h"
#include "globalvar.h"

void wavelet()
{
    int it;
    float tmp;

    for ( it=0; it<nt; it++ )
    {
        tmp = PI*f0*(it*dt-ts);
        tmp = tmp*tmp;
       // ricker[it] = (1.0-2.0*tmp)*exp(-tmp);
        ricker[it] = (it*dt-ts)*exp(-tmp);
    }

  //  // integrate the source wavelet for staggered grid use
  //  ricker[0] = ricker[0]*dt;
  //  for ( it=1; it<nt; it++ )
  //  {
  //      ricker[it] = ricker[it-1] + ricker[it]*dt;
  //  }

}
