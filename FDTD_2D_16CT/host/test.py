import torch
import torch.nn as nn
import sys
import math
from aie.utils.ml import DataShaper
import time
import os
import numpy as np
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils
import einops
import matplotlib.pyplot as plt

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

Nx = 128
block_size = 16
num_iter = 32

"""
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

"""

def generate_space(Nx: int = 128) -> torch.tensor:
    space = torch.zeros((16, Nx), dtype = torch.float32)
    space[0]  = 0          # u;
    space[1]  = 0          # last_u;
    space[2]  = 0.4        # theta;
    space[3]  = 0          # auatheta;
    space[4]  = 0          # lambda0;
    space[5]  = 0          # last_lambda0;
    space[6]  = 0          # Mom0;
    space[7]  = 0          # lambda1;
    space[8]  = 0          # last_lambda1;
    space[9]  = 0          # Mom1;
    space[10] = 0          # lambda2;
    space[11] = 0          # last_lambda2;
    space[12] = 0          # Mom2;
    space[13] = 0          # lambda3;
    space[14] = 0          # last_lambda3;
    space[15] = 0          # Mom3;

    space = einops.rearrange(space, 'p d -> d p').contiguous()

    return space




def main(opts):
    design = "TwNet_1D"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    npu_time_total = 0

    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("float32")
    dtype_src = np.dtype("float32")
    dtype_out = np.dtype("float32")

    shape_in = (Nx, 16)
    shape_src = (1, 16)
    shape_out = (Nx, 16)

    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        shape_in,
        dtype_in,
        shape_src,
        dtype_src,
        shape_out,
        dtype_out,
    )
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    int_inp = generate_space(Nx)
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------

    print(before_input.shape)
    src = 0.75 * torch.cos(torch.arange(16) * 0.03)
    print(src.shape)
    src = src.squeeze().data.numpy().astype(dtype_src)

    aie_output = before_input
    for i in range(num_iter):
        print(i)

        start = time.time_ns()
        out_temp = execute(app, aie_output, src)
        aie_output = out_temp

        stop = time.time_ns()

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time


    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = aie_output.reshape(16, Nx)
     # ofm_mem_fmt = temp_out
     # ofm_mem_fmt.tofile(
     #     log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
     # )
    # ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    print(einops.rearrange(aie_output[:, 0], '(p q) -> p q', q = block_size))
    plt.plot(aie_output[:, 0])
    plt.plot(aie_output[:, 1])
    plt.show()
    # print(aie_output[:, 0])
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total) / 1000)))


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
