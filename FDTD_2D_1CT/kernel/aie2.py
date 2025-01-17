import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape

import aie.utils.trace as trace_utils

COLS = 256
ROWS = 256
TOTAL_SIZE = COLS * ROWS
BLOCK_SIZE = COLS
BLOCKS = TOTAL_SIZE // BLOCK_SIZE

CHANNELIN0 = BLOCK_SIZE * 3
CHANNELIN1 = BLOCK_SIZE * 4
CHANNELOUT0 = BLOCK_SIZE * 3
CHANNELOUT1 = BLOCK_SIZE * 4

NT = 72

def FDTD_2D():
    @device(AIEDevice.npu1_1col)
    def device_body():

        # In
        actIn_ty = np.ndarray[(CHANNELIN0, ), np.dtype[np.float32]]
        # meta
        metaIn_ty = np.ndarray[(CHANNELIN1, ), np.dtype[np.uint8]]
        # out
        actOut_ty = np.ndarray[(CHANNELIN0, ), np.dtype[np.float32]]
        metaOut_ty = np.ndarray[(CHANNELIN1, ), np.dtype[np.uint8]]

        act_ty = np.ndarray[(BLOCK_SIZE, ), np.dtype[np.float32]]
        # AIE Core Function declarations
        FDTD_2D = external_func(
            "fdtd_2d",
            inputs=[
                actIn_ty,
                actIn_ty,
                act_ty,
                metaIn_ty,
                actOut_ty,
                metaOut_ty,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        passThrough = external_func(
            "passThrough",
            inputs=[
                actIn_ty,
                act_ty,
                actOut_ty,
            ],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        # Input
        act_In = object_fifo("act_In", ShimTile, ComputeTile2, 2, actIn_ty)
        # meta
        meta_mem_In = object_fifo("meta_mem_In", ShimTile, MemTile, 2, metaIn_ty)
        meta_In = object_fifo("meta_In", MemTile, ComputeTile2, 2, metaIn_ty)
        object_fifo_link(meta_mem_In, meta_In)

        # Output
        act_out = object_fifo("act_out", ComputeTile2, ShimTile, 2, actOut_ty)
        # meta
        meta_out = object_fifo("meta_out", ComputeTile2, MemTile, 2, metaIn_ty)
        meta_mem_out = object_fifo("meta_mem_out", MemTile, ShimTile, 2, metaIn_ty)
        object_fifo_link(meta_out, meta_mem_out)

        UPDATEDHY0 = buffer(ComputeTile2, np.ndarray[(BLOCK_SIZE, ), np.dtype[np.float32]], "UPDATEDHY0")
        Buffer0 = buffer(ComputeTile2, np.ndarray[(CHANNELIN0, ), np.dtype[np.float32]], "Buffer0")

        # Compute tile 2
        @core(ComputeTile2, 'fdtd_2d.o')
        def core_body():
            for _ in range_(0xFFFFFFFF):

                elemIn  = act_In.acquire(ObjectFifoPort.Consume, 1)
                call(
                    passThrough,
                    [
                        elemIn,
                        UPDATEDHY0,
                        Buffer0,
                    ]
                )
                objectfifo_release(ObjectFifoPort.Consume, "act_In", 1)

                for i in range_(BLOCKS - 1):
                    elemIn  = act_In.acquire(ObjectFifoPort.Consume, 1)
                    elemmetaIn = meta_In.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = act_out.acquire(ObjectFifoPort.Produce, 1)
                    elemmetaOut = meta_out.acquire(ObjectFifoPort.Produce, 1)
                    call(
                        FDTD_2D,
                        [
                            Buffer0,
                            elemIn,
                            UPDATEDHY0,
                            elemmetaIn,
                            elemOut,
                            elemmetaOut,
                            NT,
                            BLOCKS,
                            BLOCKS,
                        ]
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "act_In", 1)
                    objectfifo_release(ObjectFifoPort.Consume, "meta_In", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "act_out", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "meta_out", 1)

                elemmetaIn = meta_In.acquire(ObjectFifoPort.Consume, 1)
                elemOut = act_out.acquire(ObjectFifoPort.Produce, 1)
                elemmetaOut = meta_out.acquire(ObjectFifoPort.Produce, 1)
                call(
                    FDTD_2D,
                    [
                        Buffer0,
                        Buffer0,
                        UPDATEDHY0,
                        elemmetaIn,
                        elemOut,
                        elemmetaOut,
                        NT,
                        BLOCKS - 1,
                        BLOCKS,
                    ]
                )
                objectfifo_release(ObjectFifoPort.Consume, "meta_In", 1)
                objectfifo_release(ObjectFifoPort.Produce, "act_out", 1)
                objectfifo_release(ObjectFifoPort.Produce, "meta_out", 1)

        # To/from AIE-array data movement

        tensorIn_ty = np.ndarray[(TOTAL_SIZE * 3,), np.dtype[np.float32]]
        tensormeta_ty = np.ndarray[(TOTAL_SIZE * 4,), np.dtype[np.uint8]]

        @runtime_sequence(tensorIn_ty, tensormeta_ty)
        def sequence(I, W):
            for _ in range(4):
                npu_dma_memcpy_nd(
                    metadata=act_In,
                    bd_id=1,
                    mem=I,
                    offsets = [0, 0, 0, 0],
                    sizes=[64, ROWS, 3*1, COLS],
                    strides=[0, 3*COLS, COLS, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=act_out,
                    bd_id=0,
                    mem=I,
                    offsets = [0, 0, 0, 0],
                    sizes=[64, ROWS, 3*1, COLS],
                    strides=[0, 3*COLS, COLS, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=meta_mem_In,
                    bd_id=2,
                    mem=W,
                    offsets = [0, 0, 0, 0],
                    sizes=[64, ROWS, 4*1, COLS],
                    strides=[0, 4*COLS, COLS, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=meta_mem_out,
                    bd_id=3,
                    mem=W,
                    offsets = [0, 0, 0, 0],
                    sizes=[64, ROWS, 4*1, COLS],
                    strides=[0, 4*COLS, COLS, 1],
                )
            dma_wait(act_out, meta_mem_out)

# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    FDTD_2D()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


