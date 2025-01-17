#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "Update.h"
#include "passThrough_aie.h"

const int BLOCK_SIZE = 512;

extern "C" {

void fdtd_2d(
        float *restrict in_m,
        float *restrict in_down,
        float *restrict UpdatedHy,
        uint8_t *restrict meta,
        float *restrict out0,
        uint8_t *restrict out1,
        const int32_t NT,
        const int32_t row,
        const int32_t ROWS
) {
  Update<float, BLOCK_SIZE>(in_m, in_down, UpdatedHy, meta, out0, out1, NT, row, ROWS);
}

void passThrough(float *in, float *in1, float *out) {
  passThrough_aie<float, BLOCK_SIZE>(in, in1, out);
}
}
