#ifndef __PASSTHROUGH_AIE_H__
#define __PASSTHROUGH_AIE_H__

template <typename T, int W = 128>
void passThrough_aie(T *restrict in0, T *restrict UpdatedHy, T *restrict out) {
    event0();
    const int vec_factor = 16;

    aie::vector<T, vec_factor> In0;
    aie::vector<T, vec_factor> In1;
    aie::vector<T, vec_factor> In2;
    aie::vector<T, vec_factor> Out;
    aie::vector<T, vec_factor> zeros = aie::zeros<T, vec_factor>();

    const int F = W / vec_factor;
    for (int i = 0; i < F; i++)
        chess_prepare_for_pipelining chess_loop_range(6, ) { 
            In0 = aie::load_v<vec_factor>(in0);
            in0 += vec_factor;
            In1 = aie::load_v<vec_factor>(in0);
            in0 += vec_factor;
            In2 = aie::load_v<vec_factor>(in0);
            in0 += vec_factor;
            aie::store_v(out, In0);
            out += vec_factor;
            aie::store_v(out, In1);
            out += vec_factor;
            aie::store_v(out, In2);
            out += vec_factor;
            aie::store_v(UpdatedHy, zeros);
            UpdatedHy += vec_factor;
        }
    event1();
}
#endif
