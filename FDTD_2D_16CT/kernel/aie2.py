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

COLS = 512
ROWS = 512
TOTAL_SIZE = COLS * ROWS
BLOCK_SIZE = COLS
BLOCKS = TOTAL_SIZE // BLOCK_SIZE

CHANNELIN0 = BLOCK_SIZE * 3
CHANNELIN1 = BLOCK_SIZE * 4
CHANNELOUT0 = BLOCK_SIZE * 3
CHANNELOUT1 = BLOCK_SIZE * 4

NT = 72
NCOLS = 4
NROWS = 4

def FDTD_2D():
    @device(AIEDevice.npu1_4col)
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
        ShimTile_0_0 = tile(0, 0)
        MemTile_0_1 = tile(0, 1)
        ShimTile_3_0 = tile(3, 0)
        MemTile_3_1 = tile(3, 1)
        compute_tiles = [
            [tile(0, 2), tile(0, 3), tile(0, 4), tile(0, 5)],
            [tile(1, 5), tile(1, 4), tile(1, 3), tile(1, 2)],
            [tile(2, 2), tile(2, 3), tile(2, 4), tile(2, 5)],
            [tile(3, 5), tile(3, 4), tile(3, 3), tile(3, 2)],
        ]

        # AIE-array data movement with object fifos
        # Input
        act_In_fifos = []
        act_In_fifos.append(object_fifo(f"act_In", ShimTile_0_0, compute_tiles[0][0], 2, actIn_ty))
        act_In_fifos.append(object_fifo(f"act_In_0_2_3", compute_tiles[0][0], compute_tiles[0][1], 2, actIn_ty))
        act_In_fifos.append(object_fifo(f"act_In_0_3_4", compute_tiles[0][1], compute_tiles[0][2], 2, actIn_ty))
        act_In_fifos.append(object_fifo(f"act_In_0_4_5", compute_tiles[0][2], compute_tiles[0][3], 2, actIn_ty))

        for col in range(NCOLS - 1):
            cc = col + 1
            act_In_fifos.append(object_fifo(f"act_In_{cc}_0_3", compute_tiles[col][3], compute_tiles[cc][0], 2, actIn_ty))
            act_In_fifos.append(object_fifo(f"act_In_{cc}_2_3", compute_tiles[cc][0], compute_tiles[cc][1], 2, actIn_ty))
            act_In_fifos.append(object_fifo(f"act_In_{cc}_3_4", compute_tiles[cc][1], compute_tiles[cc][2], 2, actIn_ty))
            act_In_fifos.append(object_fifo(f"act_In_{cc}_4_5", compute_tiles[cc][2], compute_tiles[cc][3], 2, actIn_ty))

        act_In_fifos.append(object_fifo(f"act_out", compute_tiles[3][3], ShimTile_3_0, 2, actOut_ty))

        # meta
        meta_In_fifos = []
        meta_In_fifos.append(object_fifo(f"meta_mem_In", ShimTile_0_0, MemTile_0_1, 2, metaIn_ty))
        meta_In_fifos.append(object_fifo(f"meta_In", MemTile_0_1, compute_tiles[0][0], 2, metaIn_ty))

        meta_In_fifos.append(object_fifo(f"meta_In_0_2_3", compute_tiles[0][0], compute_tiles[0][1], 2, metaIn_ty))
        meta_In_fifos.append(object_fifo(f"meta_In_0_3_4", compute_tiles[0][1], compute_tiles[0][2], 2, metaIn_ty))
        meta_In_fifos.append(object_fifo(f"meta_In_0_4_5", compute_tiles[0][2], compute_tiles[0][3], 2, metaIn_ty))

        for col in range(NCOLS - 1):
            cc = col + 1
            meta_In_fifos.append(object_fifo(f"meta_In_{cc}_0_3", compute_tiles[col][3], compute_tiles[cc][0], 2, metaIn_ty))
            meta_In_fifos.append(object_fifo(f"meta_In_{cc}_2_3", compute_tiles[cc][0], compute_tiles[cc][1], 2, metaIn_ty))
            meta_In_fifos.append(object_fifo(f"meta_In_{cc}_3_4", compute_tiles[cc][1], compute_tiles[cc][2], 2, metaIn_ty))
            meta_In_fifos.append(object_fifo(f"meta_In_{cc}_4_5", compute_tiles[cc][2], compute_tiles[cc][3], 2, metaIn_ty))

        meta_In_fifos.append(object_fifo(f"meta_out", compute_tiles[3][3], MemTile_3_1, 2, metaOut_ty))
        meta_In_fifos.append(object_fifo(f"meta_mem_out", MemTile_3_1, ShimTile_3_0, 2, metaOut_ty))
        object_fifo_link(meta_In_fifos[0], meta_In_fifos[1])
        object_fifo_link(meta_In_fifos[-2], meta_In_fifos[-1])

        UPDATEDHY_list = []
        for col in range(NCOLS):
            list = []
            for row in range(NROWS):
                list.append(buffer(compute_tiles[col][row], np.ndarray[(BLOCK_SIZE, ), np.dtype[np.float32]], f"UPDATEDHY_{col}{row}"))
            UPDATEDHY_list.append(list)
            del list

        BUFFER_list = []
        for col in range(NCOLS):
            list = []
            for row in range(NROWS):
                list.append(buffer(compute_tiles[col][row], np.ndarray[(CHANNELIN0, ), np.dtype[np.float32]], f"Buffer_{col}{row}"))
            BUFFER_list.append(list)

        # Compute tile
        for col in range(NCOLS):
            for row in range(NROWS):
                @core(compute_tiles[col][row], f'fdtd_2d.o')
                def core_body():
                    for _ in range_(0xFFFFFFFF):

                        elemIn  = act_In_fifos[col*NROWS+row].acquire(ObjectFifoPort.Consume, 1)
                        call(
                            passThrough,
                            [
                                elemIn,
                                UPDATEDHY_list[col][row],
                                BUFFER_list[col][row],
                            ]
                        )
                        act_In_fifos[col*NROWS+row].release(ObjectFifoPort.Consume, 1)

                        for i in range_(BLOCKS - 1):
                            elemIn  = act_In_fifos[col*NROWS+row].acquire(ObjectFifoPort.Consume, 1)
                            elemmetaIn = meta_In_fifos[col*NROWS+row+1].acquire(ObjectFifoPort.Consume, 1)
                            elemOut = act_In_fifos[col*NROWS+row+1].acquire(ObjectFifoPort.Produce, 1)
                            elemmetaOut = meta_In_fifos[col*NROWS+row+2].acquire(ObjectFifoPort.Produce, 1)
                            call(
                                FDTD_2D,
                                [
                                    BUFFER_list[col][row],
                                    elemIn,
                                    UPDATEDHY_list[col][row],
                                    elemmetaIn,
                                    elemOut,
                                    elemmetaOut,
                                    NT,
                                    BLOCKS,
                                    BLOCKS,
                                ]
                            )
                            act_In_fifos[col*NROWS+row].release(ObjectFifoPort.Consume, 1)
                            meta_In_fifos[col*NROWS+row+1].release(ObjectFifoPort.Consume, 1)
                            act_In_fifos[col*NROWS+row+1].release(ObjectFifoPort.Produce, 1)
                            meta_In_fifos[col*NROWS+row+2].release(ObjectFifoPort.Produce, 1)

                        elemmetaIn = meta_In_fifos[col*NROWS+row+1].acquire(ObjectFifoPort.Consume, 1)
                        elemOut = act_In_fifos[col*NROWS+row+1].acquire(ObjectFifoPort.Produce, 1)
                        elemmetaOut = meta_In_fifos[col*NROWS+row+2].acquire(ObjectFifoPort.Produce, 1)
                        call(
                            FDTD_2D,
                            [
                                BUFFER_list[col][row],
                                BUFFER_list[col][row],
                                UPDATEDHY_list[col][row],
                                elemmetaIn,
                                elemOut,
                                elemmetaOut,
                                NT,
                                BLOCKS - 1,
                                BLOCKS,
                            ]
                        )
                        meta_In_fifos[col*NROWS+row+1].release(ObjectFifoPort.Consume, 1)
                        act_In_fifos[col*NROWS+row+1].release(ObjectFifoPort.Produce, 1)
                        meta_In_fifos[col*NROWS+row+2].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement

        tensorIn_ty = np.ndarray[(TOTAL_SIZE * 3,), np.dtype[np.float32]]
        tensormeta_ty = np.ndarray[(TOTAL_SIZE * 4,), np.dtype[np.uint32]]

        @runtime_sequence(tensorIn_ty, tensormeta_ty)
        def sequence(I, W):
            npu_dma_memcpy_nd(
                metadata=act_In_fifos[0],
                bd_id=1,
                mem=I,
                offsets = [0, 0, 0, 0],
                sizes=[64, ROWS, 3*1, COLS],
                strides=[0, 3*COLS, COLS, 1],
            )
            npu_dma_memcpy_nd(
                metadata=act_In_fifos[-1],
                bd_id=0,
                mem=I,
                offsets = [0, 0, 0, 0],
                sizes=[64, ROWS, 3*1, COLS],
                strides=[0, 3*COLS, COLS, 1],
            )
            npu_dma_memcpy_nd(
                metadata=meta_In_fifos[0],
                bd_id=2,
                mem=W,
                offsets = [0, 0, 0, 0],
                sizes=[64, ROWS, 1, COLS],
                strides=[0, 1*COLS, COLS, 1],
            )
            npu_dma_memcpy_nd(
                metadata=meta_In_fifos[-1],
                bd_id=3,
                mem=W,
                offsets = [0, 0, 0, 0],
                sizes=[64, ROWS, 1*1, COLS],
                strides=[0, 1*COLS, COLS, 1],
            )
            dma_wait(act_In_fifos[-1], meta_In_fifos[-1])

# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    FDTD_2D()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


