#ifndef __UPDATE_H__
#define __UPDATE_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "c_table.h"
#include "ca_table.h"
#include "src_table.h"

//*****************************************************************************
template<typename T, int W = 128>
void Update(T *restrict in_m, T *restrict in_down, T *restrict UpdatedHy, uint8_t *restrict meta, T *restrict out0, uint8_t *restrict out1, const int32_t NT, const int32_t row, const int32_t ROWS) {

    event0();
    const int vec_factor = 16;

    //C, Ca, src
    const int step = 0;
    using lut_type_float = aie::lut<4, float>;
    lut_type_float c_table(num_entries_c_table, c_table_ab, c_table_cd);
    aie::parallel_lookup<uint8_t, lut_type_float, aie::lut_oor_policy::saturate>lookup_c(c_table, step);
    lut_type_float ca_table(num_entries_ca_table, ca_table_ab, ca_table_cd);
    aie::parallel_lookup<uint8_t, lut_type_float, aie::lut_oor_policy::saturate>lookup_ca(ca_table, step);
    lut_type_float src_table(num_entries_src_table, src_table_ab, src_table_cd);
    aie::parallel_lookup<uint8_t, lut_type_float, aie::lut_oor_policy::saturate>lookup_src(src_table, step);

    aie::vector<T, vec_factor> C;
    aie::vector<T, vec_factor> Ca;
    aie::vector<T, vec_factor> src;

    //channel 1
    aie::vector<uint8_t, vec_factor> C_index;
    aie::vector<uint8_t, vec_factor> Ca_index;
    aie::vector<uint8_t, vec_factor> src_index;
    aie::vector<uint8_t, vec_factor> mask_in;

    aie::mask<vec_factor> location_status;
    const aie::vector<uint8_t, vec_factor> ones = aie::broadcast<uint8_t, vec_factor>(1);

   //channel 0
    aie::vector<T, vec_factor> Ez_in_m;
    aie::vector<T, vec_factor> Ez_in_m_r;
    aie::vector<T, vec_factor> Hx_in_m;
    aie::vector<T, vec_factor> Hy_in_m;
    aie::vector<T, vec_factor> Ez_in_down;
    aie::vector<T, vec_factor> Hx_in_down;
    aie::vector<T, vec_factor> Hy_in_down;

    //vector 
    static aie::vector<T, vec_factor> left_Hx = aie::broadcast<T, vec_factor>(0);
    const aie::vector<T, vec_factor> zeros = aie::zeros<T, vec_factor>();

    const int F = W / vec_factor;
    for (int i = 0; i < F; i++)
        chess_prepare_for_pipelining chess_loop_range(16, ){
    //{
            event0();
            //in_m 
            float *last_in_m = in_m;
            Ez_in_m = aie::load_v<vec_factor>(in_m);
            in_m += vec_factor;
            Hx_in_m = aie::load_v<vec_factor>(in_m);
            in_m += vec_factor;
            Hy_in_m = aie::load_v<vec_factor>(in_m);
            in_m += vec_factor;
            Ez_in_m_r = aie::load_v<vec_factor>(in_m);
            
            //in_down
            Ez_in_down = aie::load_v<vec_factor>(in_down);
            in_down += vec_factor;
            Hx_in_down = aie::load_v<vec_factor>(in_down);
            in_down += vec_factor; 
            Hy_in_down = aie::load_v<vec_factor>(in_down);
            in_down += vec_factor; 
            //updated Hy
            aie::vector<T, vec_factor> Hy_Updated_up = aie::load_v<vec_factor>(UpdatedHy);

            //lut and mask
            C_index = aie::load_v<vec_factor>(meta);
            meta += vec_factor;
            Ca_index = aie::load_v<vec_factor>(meta);
            meta += vec_factor;
            if(*meta == NT){
                src_index = aie::broadcast<uint8_t, vec_factor>(0);
            }
            else{
                src_index = aie::load_v<vec_factor>(meta);
            }
            meta += vec_factor;
            mask_in = aie::load_v<vec_factor>(meta);
            meta += vec_factor;

            //LUT: C, Ca
            C = lookup_c.fetch(C_index);
            Ca = lookup_ca.fetch(Ca_index);
            src = lookup_src.fetch(src_index);
            aie::vector<T, vec_factor> source = aie::mul(src, (float)10);

            //mask
            location_status = aie::eq(mask_in, ones);

            // update
            // C * Hx, C * Hy, C * Ez
            //aie::vector<T, vec_factor> p1_Hx = aie::mul(C, Hx_in_m);
            //aie::vector<T, vec_factor> p1_Hy = aie::mul(C, Hy_in_m);
            //aie::vector<T, vec_factor> p1_Ez = aie::mul(C, Ez_in_m);
            //Update Hy
            aie::vector<T, vec_factor> p2_Hy_Ez;
            if (row == (ROWS - 1)){
                p2_Hy_Ez = aie::sub(zeros, Ez_in_m);
            }
            else{
                p2_Hy_Ez = aie::sub(Ez_in_down, Ez_in_m);
            }
            aie::vector<T, vec_factor> p3_Hy = aie::add(Hy_in_m, p2_Hy_Ez);
            aie::vector<T, vec_factor> Out_Hy = aie::mul(C, p3_Hy);
            //Update Hx
            aie::vector<T, vec_factor> Ez_down;
            if(i == (F - 1)){
                Ez_down = aie::shuffle_down_fill(Ez_in_m, zeros, 1);
            }
            else{
                Ez_down = aie::shuffle_down_fill(Ez_in_m, Ez_in_m_r, 1);
            }

            aie::vector<T, vec_factor> p2_Hx_Ez = aie::sub(Ez_down, Ez_in_m);
            aie::vector<T, vec_factor> p3_Hx = aie::sub(Hx_in_m, p2_Hx_Ez);
            aie::vector<T, vec_factor> Out_Hx = aie::mul(C, p3_Hx);

            //Update Ez
            aie::vector<T, vec_factor> Hx_up;
            if(i == 0){
                Hx_up = aie::shuffle_up_fill(Out_Hx, zeros, 1);
            }
            else{
                Hx_up = aie::shuffle_up_fill(Out_Hx, left_Hx, 1);
            }
            aie::vector<T, vec_factor> p2_Ez_Hx = aie::sub(Out_Hx, Hx_up);
            aie::vector<T, vec_factor> p2_Ez_Hy = aie::sub(Out_Hy, Hy_Updated_up);
            aie::vector<T, vec_factor> p3_Ez = aie::sub(p2_Ez_Hy, p2_Ez_Hx);
            aie::vector<T, vec_factor> p4_Ez = aie::mul(Ca, p3_Ez);
            aie::vector<T, vec_factor> p5_Ez = aie::add(p4_Ez, Ez_in_m);
            aie::vector<T, vec_factor> Out_Ez = aie::mul(C, p5_Ez);

            left_Hx = Out_Hx;

            //add src
            aie::vector<T, vec_factor> Ez_out = aie::select(Out_Ez, source, location_status);
            //Out0
            //Ez
            aie::store_v(out0, Ez_out);
            out0 += vec_factor;
            //Hx
            aie::store_v(out0, Out_Hx);
            out0 += vec_factor;
            //Hy
            aie::store_v(out0, Out_Hy);
            out0 += vec_factor;
            //out1
            //lut_index
            aie::store_v(out1, C_index);
            out1 += vec_factor;
            aie::store_v(out1, Ca_index);
            out1 += vec_factor;
            //src_index + 1
            aie::vector<uint8_t, vec_factor> new_src_index = aie::add(src_index, ones);
            aie::store_v(out1, new_src_index);
            out1 += vec_factor;
            //mask_index
            aie::store_v(out1, mask_in);
            out1 += vec_factor;
            //UpdateHy
            aie::store_v(UpdatedHy, Out_Hy);
            UpdatedHy += vec_factor;
            //line1 --> line0
            aie::store_v(last_in_m, Ez_in_down);
            last_in_m += vec_factor;
            aie::store_v(last_in_m, Hx_in_down);
            last_in_m += vec_factor;
            aie::store_v(last_in_m, Hy_in_down);
        }
    event1();
}
#endif
