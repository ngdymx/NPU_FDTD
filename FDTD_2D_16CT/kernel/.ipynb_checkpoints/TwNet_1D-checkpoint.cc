#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "common.h"
#include "TwNet_1D_forward.h"
#include "Apply_u_src.h"


extern "C" {

void TwNet_1D(dtype *restrict in_l, dtype *restrict in_r, dtype *restrict src, dtype *restrict out, const metatype gamma, const metatype beta, const int32_t blocks, const int32_t block_location, const int32_t src_point) {
  dtype buffer[128];
  TwNet_1D_forward<8>((dtype *)in_l, (dtype *)in_r, (dtype *)buffer, blocks);
  Apply_u_src<8>((dtype *)buffer, (dtype *)src, (dtype *)out, block_location, src_point, gamma, beta);
}

}
