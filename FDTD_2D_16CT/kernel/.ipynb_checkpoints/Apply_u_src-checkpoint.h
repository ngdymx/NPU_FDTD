#ifndef __APPLY_U_SRC_H__
#define __APPLY_U_SRC_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "common.h"

//*****************************************************************************
template<int WIDTH = 8>
void Apply_u_src(dtype *restrict in, dtype *restrict src, dtype *restrict out, const int32_t block_location, const int32_t src_point, const metatype gamma, const metatype beta) {
        
    cell* cell_in = (cell*) in;
    cell* cell_out = (cell*) out;

    for (int x = 0; x < WIDTH; x++){
        cell_out[x].last_u = cell_in[x].last_u;
        cell_out[0].lambda0 = cell_in[0].lambda0;
        cell_out[0].lambda1 = cell_in[0].lambda1;
        cell_out[0].lambda2 = cell_in[0].lambda2;
        cell_out[0].lambda3 = cell_in[0].lambda3;
        cell_out[0].last_lambda0 = cell_in[0].last_lambda0;
        cell_out[0].last_lambda1 = cell_in[0].last_lambda1;
        cell_out[0].last_lambda2 = cell_in[0].last_lambda2;
        cell_out[0].last_lambda3 = cell_in[0].last_lambda3;
        cell_out[0].Mom0 = cell_in[0].Mom0;
        cell_out[0].Mom1 = cell_in[0].Mom1;
        cell_out[0].Mom2 = cell_in[0].Mom2;
        cell_out[0].Mom3 = cell_in[0].Mom3;
        cell_out[x].theta = cell_in[x].theta;
        cell_out[x].auatheta = cell_in[x].auatheta;
        
        if((x + block_location) == src_point){
            cell_out[x].u = *src;
        }
        else{
            cell_out[x].u = cell_in[x].u;
        }
    }
}
#endif