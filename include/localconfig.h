#ifndef LOCALCONFIG_H
#define LOCALCONFIG_H

#define    PI    (3.14159265358979)

#include <math.h>

#define  dims     2

#define  DB       2 // [修改] 原为 3

#define  KAI      (2*DB*DB)

#define  LY       2 // [修改] 原为 4

#define  LX       2 // [修改] 原为 4


#define PSONUM   12 // [修改] 原为 84 

#define THREADS_NUM  24 // [修改] 原为 90 

#define SUBSWARMS    4 // [修改] 原为 12 

#define GPNUM   (PSONUM/SUBSWARMS)

#define  ACCURACYCUT     1.0e-5


#endif
