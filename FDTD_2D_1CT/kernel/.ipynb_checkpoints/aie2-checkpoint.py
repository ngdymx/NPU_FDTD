import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

SIZE_1 = 1
SIZE_8 = 8
SIZE_16 = 16
SIZE_32 = 32
SIZE_128 = 128
SIZE_1024 = 1024

def TwNet_1D_Scalar():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            actIn_ty = T.memref(SIZE_8, T.f32())
            srcIn_ty = T.memref(SIZE_1, T.f32())

            actOut_ty = T.memref(SIZE_8, T.f32())

            # AIE Core Function declarations
            TwNet_1D = external_func(
                "TwNet_1D",
                inputs=[
                    actIn_ty,
                    actIn_ty,
                    srcIn_ty,
                    actOut_ty,
                    T.f32(),
                    T.f32(),
                    T.i32(),
                    T.i32(),
                    T.i32(),
                ],
            )

            # Tile declarations

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            # Input
            act_In = object_fifo("act_In", ShimTile, ComputeTile2, 2, actIn_ty)
            src_In = object_fifo("src_In", ShimTile, ComputeTile2, 2, srcIn_ty)

            # Output
            act_out = object_fifo("act_out", ComputeTile2, ShimTile, 2, actOut_ty)


            # Compute tile 2
            @core(ComputeTile2, "TwNet_1D.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    gamma = 0.99
                    beta = 0.5
                    block = 16
                    src_point = 15
                    srcIn = src_In.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(SIZE_16):
                        elemIn = act_In.acquire(ObjectFifoPort.Consume, 2)
                        elemOut = act_out.acquire(ObjectFifoPort.Produce, 1)

                        call(
                            TwNet_1D,
                            [
                                elemIn[0],
                                elemIn[1],
                                srcIn,
                                elemOut,
                                gamma,
                                beta,
                                block,
                                arith.index_cast(i, to=T.i32()) * 8,
                                src_point,
                                
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Produce, "act_out", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "act_In", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "src_In", 1)
                    yield_([])

            # To/from AIE-array data movement

            tensor_ty = T.memref(SIZE_2048, T.f32())
            meta_ty = T.memref(SIZE_1,  T.f32())

            @FuncOp.from_py_func(tensor_ty, meta_ty, tensor_ty)
            def sequence(I, W, O):

                npu_dma_memcpy_nd(
                    metadata="act_In",
                    bd_id=0,
                    mem=I,
                    sizes=[1, 1, 1, SIZE_2048],
                )
                npu_dma_memcpy_nd(
                    metadata="act_out",
                    bd_id=2,
                    mem=O,
                    sizes=[1, 1, 1, SIZE_2048],
                )
                npu_dma_memcpy_nd(
                    metadata="src_In",
                    bd_id=1,
                    mem=W,
                    offsets = [0, 0, 0, 0],
                    sizes=[1, 1, 1, SIZE_1],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


TwNet_1D_Scalar()
