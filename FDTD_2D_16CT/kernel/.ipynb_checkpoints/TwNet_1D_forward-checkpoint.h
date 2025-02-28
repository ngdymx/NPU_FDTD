#ifndef __TWNET_1D_H__
#define __TWNET_1D_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "common.h"

//*****************************************************************************
template<int W = 8>
void TwNet_1D_forward(dtype *restrict in_l, dtype *restrict in_r, dtype *restrict out, const int32_t blocks) {

    static int block_counter = 0;
    static savetokernel left_edge;

    cell* cell_in_l = (cell*) in_l;
    cell* cell_in_r = (cell*) in_r;
    cell* cell_out = (cell*) out;

    bool is_left = block_counter == 0;
    bool is_right = block_counter == (blocks - 1);
    block_counter = is_right? 0 : (block_counter + 1);

    if (is_left){
        cell_out[0].last_u = cell_out[0].u;
        cell_out[0].u = (2 - 2 * cell_in_l[0].theta) * cell_in_l[0].u + cell_in_l[0].theta * cell_in_l[1].u - cell_in_l[0].last_u;
        cell_out[0].auatheta = cell_in_l[1].u - 2 * cell_in_l[0].u;
        
        cell_out[0].theta = cell_in_l[0].theta;
        cell_out[0].lambda0 = cell_in_l[0].lambda0;
        cell_out[0].lambda1 = cell_in_l[0].lambda1;
        cell_out[0].lambda2 = cell_in_l[0].lambda2;
        cell_out[0].lambda3 = cell_in_l[0].lambda3;
        cell_out[0].last_lambda0 = cell_in_l[0].last_lambda0;
        cell_out[0].last_lambda1 = cell_in_l[0].last_lambda1;
        cell_out[0].last_lambda2 = cell_in_l[0].last_lambda2;
        cell_out[0].last_lambda3 = cell_in_l[0].last_lambda3;
        cell_out[0].Mom0 = cell_in_l[0].Mom0;
        cell_out[0].Mom1 = cell_in_l[0].Mom1;
        cell_out[0].Mom2 = cell_in_l[0].Mom2;
        cell_out[0].Mom3 = cell_in_l[0].Mom3;
    }
    else{
        cell_out[0].last_u = cell_out[0].u;
        cell_out[0].u = (2 - 2 * cell_in_l[0].theta) * cell_in_l[0].u + cell_in_l[0].theta * (cell_in_l[1].u + left_edge.u) - cell_in_l[0].last_u;
        cell_out[0].auatheta = cell_in_l[1].u + left_edge.u - 2 * cell_in_l[0].u;
        
        cell_out[0].theta = cell_in_l[0].theta;
        cell_out[0].lambda0 = cell_in_l[0].lambda0;
        cell_out[0].lambda1 = cell_in_l[0].lambda1;
        cell_out[0].lambda2 = cell_in_l[0].lambda2;
        cell_out[0].lambda3 = cell_in_l[0].lambda3;
        cell_out[0].last_lambda0 = cell_in_l[0].last_lambda0;
        cell_out[0].last_lambda1 = cell_in_l[0].last_lambda1;
        cell_out[0].last_lambda2 = cell_in_l[0].last_lambda2;
        cell_out[0].last_lambda3 = cell_in_l[0].last_lambda3;
        cell_out[0].Mom0 = cell_in_l[0].Mom0;
        cell_out[0].Mom1 = cell_in_l[0].Mom1;
        cell_out[0].Mom2 = cell_in_l[0].Mom2;
        cell_out[0].Mom3 = cell_in_l[0].Mom3;
    }
    left_edge.u = cell_in_l[W - 1].u;
    left_edge.theta = cell_in_l[W - 1].theta;
    left_edge.lambda0 = cell_in_l[W - 1].lambda0;
    left_edge.lambda1 = cell_in_l[W - 1].lambda1;

    for (int x = 1; x < W - 1; x++) { // col of output image
        cell_out[x].last_u = cell_out[x].u;
        cell_out[x].u = (2 - 2 * cell_in_l[x].theta) * cell_in_l[x].u + cell_in_l[x].theta * (cell_in_l[x - 1].u + cell_in_l[x + 1].u) - cell_in_l[x].last_u;
        cell_out[x].auatheta = cell_in_l[x - 1].u + cell_in_l[x + 1].u - 2 * cell_in_l[0].u;
        
        cell_out[x].theta = cell_in_l[x].theta;
        cell_out[0].lambda0 = cell_in_l[0].lambda0;
        cell_out[0].lambda1 = cell_in_l[0].lambda1;
        cell_out[0].lambda2 = cell_in_l[0].lambda2;
        cell_out[0].lambda3 = cell_in_l[0].lambda3;
        cell_out[0].last_lambda0 = cell_in_l[0].last_lambda0;
        cell_out[0].last_lambda1 = cell_in_l[0].last_lambda1;
        cell_out[0].last_lambda2 = cell_in_l[0].last_lambda2;
        cell_out[0].last_lambda3 = cell_in_l[0].last_lambda3;
        cell_out[0].Mom0 = cell_in_l[0].Mom0;
        cell_out[0].Mom1 = cell_in_l[0].Mom1;
        cell_out[0].Mom2 = cell_in_l[0].Mom2;
        cell_out[0].Mom3 = cell_in_l[0].Mom3;
    }

    if (is_right){
        cell_out[W - 1].last_u = cell_out[W - 1].u;
        cell_out[W - 1].u = (2 - 2 * cell_in_l[W - 1].theta) * cell_in_l[W - 1].u + cell_in_l[W - 1].theta * cell_in_l[W - 2].u - cell_in_l[W - 1].last_u; 
        cell_out[W - 1].auatheta = cell_in_l[W - 2].u - 2 * cell_in_l[0].u;
        
        cell_out[W - 1].theta = cell_in_l[W - 1].theta;
        cell_out[0].lambda0 = cell_in_l[0].lambda0;
        cell_out[0].lambda1 = cell_in_l[0].lambda1;
        cell_out[0].lambda2 = cell_in_l[0].lambda2;
        cell_out[0].lambda3 = cell_in_l[0].lambda3;
        cell_out[0].last_lambda0 = cell_in_l[0].last_lambda0;
        cell_out[0].last_lambda1 = cell_in_l[0].last_lambda1;
        cell_out[0].last_lambda2 = cell_in_l[0].last_lambda2;
        cell_out[0].last_lambda3 = cell_in_l[0].last_lambda3;
        cell_out[0].Mom0 = cell_in_l[0].Mom0;
        cell_out[0].Mom1 = cell_in_l[0].Mom1;
        cell_out[0].Mom2 = cell_in_l[0].Mom2;
        cell_out[0].Mom3 = cell_in_l[0].Mom3;
    }
    else{
        cell_out[W - 1].last_u = cell_out[W - 1].u;
        cell_out[W - 1].u = (2 - 2 * cell_in_l[W - 1].theta) * cell_in_l[W - 1].u + cell_in_l[W - 1].theta * (cell_in_l[W - 2].u + cell_in_r[0].u) - cell_in_l[W - 1].last_u;       
        cell_out[W - 1].auatheta = cell_in_l[W - 2].u + cell_in_r[0].u - 2 * cell_in_l[0].u;
        
        cell_out[W - 1].theta = cell_in_l[W - 1].theta;
        cell_out[0].lambda0 = cell_in_l[0].lambda0;
        cell_out[0].lambda1 = cell_in_l[0].lambda1;
        cell_out[0].lambda2 = cell_in_l[0].lambda2;
        cell_out[0].lambda3 = cell_in_l[0].lambda3;
        cell_out[0].last_lambda0 = cell_in_l[0].last_lambda0;
        cell_out[0].last_lambda1 = cell_in_l[0].last_lambda1;
        cell_out[0].last_lambda2 = cell_in_l[0].last_lambda2;
        cell_out[0].last_lambda3 = cell_in_l[0].last_lambda3;
        cell_out[0].Mom0 = cell_in_l[0].Mom0;
        cell_out[0].Mom1 = cell_in_l[0].Mom1;
        cell_out[0].Mom2 = cell_in_l[0].Mom2;
        cell_out[0].Mom3 = cell_in_l[0].Mom3;
    }
}
#endif