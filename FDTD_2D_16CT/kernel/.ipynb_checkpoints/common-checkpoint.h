#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

typedef float dtype;
typedef float metatype;

typedef struct{
    dtype u;
    dtype last_u;
    dtype theta;
    dtype auatheta;
    dtype lambda0;
    dtype last_lambda0;
    dtype Mom0;
    dtype lambda1;
    dtype last_lambda1;
    dtype Mom1;
    dtype lambda2;
    dtype last_lambda2;
    dtype Mom2;
    dtype lambda3;
    dtype last_lambda3;
    dtype Mom3;
} cell; //10*32

typedef struct{
    dtype u;
    dtype theta;
}cellOut;

typedef struct{
    dtype u;
    dtype theta;
    dtype lambda0;
    dtype lambda1;
}savetokernel;

#endif