#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "common.h"

//*****************************************************************************
template<int WIDTH = 8, SIZE>
void Apply_lambda_src(cell *restrict in, cell *restrict golden, mask_type *restrict mask, cell *restrict out, 
const metatype lr, const int32_t T, const int32_t phase_num, const int32_t t_start) {

    static int T_counter = 0;
    static int cell_counter = 0;
    cell_counter = (SIZE - 1)? 0: (cell_counter+1);
    if (cell_counter == (SIZE - 1)){
        T_counter += 1;
        if(T_counter == (T - 1)){
            T_counter = 0;
        }
        else{
            T_counter = T_counter;      
        }
    }
    else{
        T_counter = T_counter;
        if(T_counter == (T - 1)){
            T_counter = 0;
        }
        else{
            T_counter = T_counter;      
        }
    }
    
        
    cell* cell_in = (cell*) in;
    cell* cell_out = (cell*) out;
    mask_type* cell_mask = (mask_type*) mask;

    dtype* lambda_src0;
    dtype* lambda_src1;

    for (int x = 0; x < WIDTH; x++){
        cell_out[x].u = cell_in[x].u;
        cell_out[x].last_u = cell_in[x].last_u;
        cell_out[x].last_lambda0 = cell_in[x].last_lambda0;
        cell_out[x].last_lambda1 = cell_in[x].last_lambda1;
        cell_out[x].auatheta = cell_in[x].auatheta;
        if(T_counter > t_start){
            if(T_counter % T - phase_num == 0){
                if(cell_mask[e_bit_is_adsrc]){
                    lambda_src0 = cell_in[x].u - *golden;
                }
                else{
                    lambda_src0 = lambda_src0;
                }
                cell_out[x].lambda0 = cell_in[x].lambda0 + lambda_src0;
                cell_out[x].Mom0 = cell_in[x].Mom0 - lr * cell_out[x].lambda0 * cell_in[x].auatheta;
                if(cell_mask[e_bit_is_train]){
                    cell_out[x].theta = cell_in[x].theta + cell_out[x].Mom0;
                }
                else{
                    cell_out[x].theta = cell_in[x].theta;
                }
            }
            
            else if(T_counter % T - phase_num == T/4){
                if(cell_mask[e_bit_is_adsrc]){
                    lambda_src1 = cell_in[x].u - *golden;
                }
                else{
                    lambda_src1 = lambda_src1;
                }
                cell_out[x].lambda1 = cell_in[x].lambda1 + lambda_src1;
                cell_out[x].Mom1 = cell_in[x].Mom1 - lr * cell_out[x].lambda1 * cell_in[x].auatheta;
                if(cell_mask[e_bit_is_train]){
                    cell_out[x].theta = cell_in[x].theta + cell_out[x].Mom1;
                }
                else{
                    cell_out[x].theta = cell_in[x].theta;
                }
            }
        }
    }
}
