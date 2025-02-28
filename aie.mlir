module {
  aie.device(npu1_4col) {
    func.func private @fdtd_2d(memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32)
    func.func private @passThrough(memref<1536xf32>, memref<512xf32>, memref<1536xf32>)

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_3_0 = aie.tile(3, 0)
    %tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)

    aie.objectfifo @act_In(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_mem_In(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @meta_In(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo.link [@meta_mem_In] -> [@meta_In]()
    aie.objectfifo @act_02_03(%tile_0_2, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_02_03(%tile_0_2, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_03_04(%tile_0_3, {%tile_0_4}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_03_04(%tile_0_3, {%tile_0_4}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_04_05(%tile_0_4, {%tile_0_5}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_04_05(%tile_0_4, {%tile_0_5}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_05_15(%tile_0_5, {%tile_1_5}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_05_15(%tile_0_5, {%tile_1_5}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_15_14(%tile_1_5, {%tile_1_4}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_15_14(%tile_1_5, {%tile_1_4}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_14_13(%tile_1_4, {%tile_1_3}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_14_13(%tile_1_4, {%tile_1_3}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_13_12(%tile_1_3, {%tile_1_2}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_13_12(%tile_1_3, {%tile_1_2}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_12_22(%tile_1_2, {%tile_2_2}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_12_22(%tile_1_2, {%tile_2_2}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_22_23(%tile_2_2, {%tile_2_3}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_22_23(%tile_2_2, {%tile_2_3}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_23_24(%tile_2_3, {%tile_2_4}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_23_24(%tile_2_3, {%tile_2_4}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_24_25(%tile_2_4, {%tile_2_5}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_24_25(%tile_2_4, {%tile_2_5}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_25_35(%tile_2_5, {%tile_3_5}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_25_35(%tile_2_5, {%tile_3_5}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_35_34(%tile_3_5, {%tile_3_4}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_35_34(%tile_3_5, {%tile_3_4}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_34_33(%tile_3_4, {%tile_3_3}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_34_33(%tile_3_4, {%tile_3_3}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_33_32(%tile_3_3, {%tile_3_2}, 4 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_33_32(%tile_3_3, {%tile_3_2}, 4 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @act_out(%tile_3_2, {%tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1536xf32>>
    aie.objectfifo @meta_out(%tile_3_2, {%tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo @meta_mem_out(%tile_3_1, {%tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1536xui8>>
    aie.objectfifo.link [@meta_out] -> [@meta_mem_out]()

    %UPDATEDHY0 = aie.buffer(%tile_0_2) {sym_name = "UPDATEDHY0"} : memref<512xf32> 
    %Buffer0 = aie.buffer(%tile_0_2) {sym_name = "Buffer0"} : memref<1536xf32> 
    %UPDATEDHY1 = aie.buffer(%tile_0_3) {sym_name = "UPDATEDHY1"} : memref<512xf32> 
    %Buffer1 = aie.buffer(%tile_0_3) {sym_name = "Buffer1"} : memref<1536xf32> 
    %UPDATEDHY2 = aie.buffer(%tile_0_4) {sym_name = "UPDATEDHY2"} : memref<512xf32> 
    %Buffer2 = aie.buffer(%tile_0_4) {sym_name = "Buffer2"} : memref<1536xf32> 
    %UPDATEDHY3 = aie.buffer(%tile_0_5) {sym_name = "UPDATEDHY3"} : memref<512xf32> 
    %Buffer3 = aie.buffer(%tile_0_5) {sym_name = "Buffer3"} : memref<1536xf32> 
    %UPDATEDHY4 = aie.buffer(%tile_1_5) {sym_name = "UPDATEDHY4"} : memref<512xf32> 
    %Buffer4 = aie.buffer(%tile_1_5) {sym_name = "Buffer4"} : memref<1536xf32> 
    %UPDATEDHY5 = aie.buffer(%tile_1_4) {sym_name = "UPDATEDHY5"} : memref<512xf32> 
    %Buffer5 = aie.buffer(%tile_1_4) {sym_name = "Buffer5"} : memref<1536xf32> 
    %UPDATEDHY6 = aie.buffer(%tile_1_3) {sym_name = "UPDATEDHY6"} : memref<512xf32> 
    %Buffer6 = aie.buffer(%tile_1_3) {sym_name = "Buffer6"} : memref<1536xf32> 
    %UPDATEDHY7 = aie.buffer(%tile_1_2) {sym_name = "UPDATEDHY7"} : memref<512xf32> 
    %Buffer7 = aie.buffer(%tile_1_2) {sym_name = "Buffer7"} : memref<1536xf32> 
    %UPDATEDHY8 = aie.buffer(%tile_2_2) {sym_name = "UPDATEDHY8"} : memref<512xf32> 
    %Buffer8 = aie.buffer(%tile_2_2) {sym_name = "Buffer8"} : memref<1536xf32> 
    %UPDATEDHY9 = aie.buffer(%tile_2_3) {sym_name = "UPDATEDHY9"} : memref<512xf32> 
    %Buffer9 = aie.buffer(%tile_2_3) {sym_name = "Buffer9"} : memref<1536xf32> 
    %UPDATEDHY10 = aie.buffer(%tile_2_4) {sym_name = "UPDATEDHY10"} : memref<512xf32> 
    %Buffer10 = aie.buffer(%tile_2_4) {sym_name = "Buffer10"} : memref<1536xf32> 
    %UPDATEDHY11 = aie.buffer(%tile_2_5) {sym_name = "UPDATEDHY11"} : memref<512xf32> 
    %Buffer11 = aie.buffer(%tile_2_5) {sym_name = "Buffer11"} : memref<1536xf32> 
    %UPDATEDHY12 = aie.buffer(%tile_3_5) {sym_name = "UPDATEDHY12"} : memref<512xf32> 
    %Buffer12 = aie.buffer(%tile_3_5) {sym_name = "Buffer12"} : memref<1536xf32> 
    %UPDATEDHY13 = aie.buffer(%tile_3_4) {sym_name = "UPDATEDHY13"} : memref<512xf32> 
    %Buffer13 = aie.buffer(%tile_3_4) {sym_name = "Buffer13"} : memref<1536xf32> 
    %UPDATEDHY14 = aie.buffer(%tile_3_3) {sym_name = "UPDATEDHY14"} : memref<512xf32> 
    %Buffer14 = aie.buffer(%tile_3_3) {sym_name = "Buffer14"} : memref<1536xf32> 
    %UPDATEDHY15 = aie.buffer(%tile_3_2) {sym_name = "UPDATEDHY15"} : memref<512xf32> 
    %Buffer15 = aie.buffer(%tile_3_2) {sym_name = "Buffer15"} : memref<1536xf32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_In(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY0, %Buffer0) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_In(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_In(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_In(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_02_03(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_02_03(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer0, %9, %UPDATEDHY0, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_In(Consume, 1)
          aie.objectfifo.release @meta_In(Consume, 1)
          aie.objectfifo.release @act_02_03(Produce, 1)
          aie.objectfifo.release @meta_02_03(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_In(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_02_03(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_02_03(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer0, %Buffer0, %UPDATEDHY0, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_In(Consume, 1)
        aie.objectfifo.release @act_02_03(Produce, 1)
        aie.objectfifo.release @meta_02_03(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_02_03(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY1, %Buffer1) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_02_03(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_02_03(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_02_03(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_03_04(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_03_04(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer1, %9, %UPDATEDHY1, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_02_03(Consume, 1)
          aie.objectfifo.release @meta_02_03(Consume, 1)
          aie.objectfifo.release @act_03_04(Produce, 1)
          aie.objectfifo.release @meta_03_04(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_02_03(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_03_04(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_03_04(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer1, %Buffer1, %UPDATEDHY1, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_02_03(Consume, 1)
        aie.objectfifo.release @act_03_04(Produce, 1)
        aie.objectfifo.release @meta_03_04(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_03_04(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY2, %Buffer2) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_03_04(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_03_04(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_03_04(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_04_05(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_04_05(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer2, %9, %UPDATEDHY2, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_03_04(Consume, 1)
          aie.objectfifo.release @meta_03_04(Consume, 1)
          aie.objectfifo.release @act_04_05(Produce, 1)
          aie.objectfifo.release @meta_04_05(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_03_04(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_04_05(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_04_05(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer2, %Buffer2, %UPDATEDHY2, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_03_04(Consume, 1)
        aie.objectfifo.release @act_04_05(Produce, 1)
        aie.objectfifo.release @meta_04_05(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_04_05(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY3, %Buffer3) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_04_05(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_04_05(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_04_05(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_05_15(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_05_15(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer3, %9, %UPDATEDHY3, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_04_05(Consume, 1)
          aie.objectfifo.release @meta_04_05(Consume, 1)
          aie.objectfifo.release @act_05_15(Produce, 1)
          aie.objectfifo.release @meta_05_15(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_04_05(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_05_15(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_05_15(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer3, %Buffer3, %UPDATEDHY3, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_04_05(Consume, 1)
        aie.objectfifo.release @act_05_15(Produce, 1)
        aie.objectfifo.release @meta_05_15(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_05_15(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY4, %Buffer4) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_05_15(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_05_15(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_05_15(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_15_14(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_15_14(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer4, %9, %UPDATEDHY4, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_05_15(Consume, 1)
          aie.objectfifo.release @meta_05_15(Consume, 1)
          aie.objectfifo.release @act_15_14(Produce, 1)
          aie.objectfifo.release @meta_15_14(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_05_15(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_15_14(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_15_14(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer4, %Buffer4, %UPDATEDHY4, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_05_15(Consume, 1)
        aie.objectfifo.release @act_15_14(Produce, 1)
        aie.objectfifo.release @meta_15_14(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_15_14(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY5, %Buffer5) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_15_14(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_15_14(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_15_14(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_14_13(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_14_13(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer5, %9, %UPDATEDHY5, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_15_14(Consume, 1)
          aie.objectfifo.release @meta_15_14(Consume, 1)
          aie.objectfifo.release @act_14_13(Produce, 1)
          aie.objectfifo.release @meta_14_13(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_15_14(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_14_13(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_14_13(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer5, %Buffer5, %UPDATEDHY5, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_15_14(Consume, 1)
        aie.objectfifo.release @act_14_13(Produce, 1)
        aie.objectfifo.release @meta_14_13(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_14_13(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY6, %Buffer6) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_14_13(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_14_13(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_14_13(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_13_12(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_13_12(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer6, %9, %UPDATEDHY6, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_14_13(Consume, 1)
          aie.objectfifo.release @meta_14_13(Consume, 1)
          aie.objectfifo.release @act_13_12(Produce, 1)
          aie.objectfifo.release @meta_13_12(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_14_13(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_13_12(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_13_12(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer6, %Buffer6, %UPDATEDHY6, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_14_13(Consume, 1)
        aie.objectfifo.release @act_13_12(Produce, 1)
        aie.objectfifo.release @meta_13_12(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_13_12(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY7, %Buffer7) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_13_12(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_13_12(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_13_12(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_12_22(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_12_22(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer7, %9, %UPDATEDHY7, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_13_12(Consume, 1)
          aie.objectfifo.release @meta_13_12(Consume, 1)
          aie.objectfifo.release @act_12_22(Produce, 1)
          aie.objectfifo.release @meta_12_22(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_13_12(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_12_22(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_12_22(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer7, %Buffer7, %UPDATEDHY7, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_13_12(Consume, 1)
        aie.objectfifo.release @act_12_22(Produce, 1)
        aie.objectfifo.release @meta_12_22(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_12_22(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY8, %Buffer8) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_12_22(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_12_22(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_12_22(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_22_23(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_22_23(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer8, %9, %UPDATEDHY8, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_12_22(Consume, 1)
          aie.objectfifo.release @meta_12_22(Consume, 1)
          aie.objectfifo.release @act_22_23(Produce, 1)
          aie.objectfifo.release @meta_22_23(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_12_22(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_22_23(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_22_23(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer8, %Buffer8, %UPDATEDHY8, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_12_22(Consume, 1)
        aie.objectfifo.release @act_22_23(Produce, 1)
        aie.objectfifo.release @meta_22_23(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_22_23(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY9, %Buffer9) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_22_23(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_22_23(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_22_23(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_23_24(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_23_24(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer9, %9, %UPDATEDHY9, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_22_23(Consume, 1)
          aie.objectfifo.release @meta_22_23(Consume, 1)
          aie.objectfifo.release @act_23_24(Produce, 1)
          aie.objectfifo.release @meta_23_24(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_22_23(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_23_24(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_23_24(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer9, %Buffer9, %UPDATEDHY9, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_22_23(Consume, 1)
        aie.objectfifo.release @act_23_24(Produce, 1)
        aie.objectfifo.release @meta_23_24(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_23_24(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY10, %Buffer10) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_23_24(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_23_24(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_23_24(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_24_25(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_24_25(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer10, %9, %UPDATEDHY10, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_23_24(Consume, 1)
          aie.objectfifo.release @meta_23_24(Consume, 1)
          aie.objectfifo.release @act_24_25(Produce, 1)
          aie.objectfifo.release @meta_24_25(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_23_24(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_24_25(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_24_25(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer10, %Buffer10, %UPDATEDHY10, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_23_24(Consume, 1)
        aie.objectfifo.release @act_24_25(Produce, 1)
        aie.objectfifo.release @meta_24_25(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_24_25(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY11, %Buffer11) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_24_25(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_24_25(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_24_25(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_25_35(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_25_35(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer11, %9, %UPDATEDHY11, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_24_25(Consume, 1)
          aie.objectfifo.release @meta_24_25(Consume, 1)
          aie.objectfifo.release @act_25_35(Produce, 1)
          aie.objectfifo.release @meta_25_35(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_24_25(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_25_35(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_25_35(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer11, %Buffer11, %UPDATEDHY11, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_24_25(Consume, 1)
        aie.objectfifo.release @act_25_35(Produce, 1)
        aie.objectfifo.release @meta_25_35(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_25_35(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY12, %Buffer12) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_25_35(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_25_35(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_25_35(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_35_34(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_35_34(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer12, %9, %UPDATEDHY12, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_25_35(Consume, 1)
          aie.objectfifo.release @meta_25_35(Consume, 1)
          aie.objectfifo.release @act_35_34(Produce, 1)
          aie.objectfifo.release @meta_35_34(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_25_35(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_35_34(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_35_34(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer12, %Buffer12, %UPDATEDHY12, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_25_35(Consume, 1)
        aie.objectfifo.release @act_35_34(Produce, 1)
        aie.objectfifo.release @meta_35_34(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_35_34(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY13, %Buffer13) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_35_34(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_35_34(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_35_34(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_34_33(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_34_33(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer13, %9, %UPDATEDHY13, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_35_34(Consume, 1)
          aie.objectfifo.release @meta_35_34(Consume, 1)
          aie.objectfifo.release @act_34_33(Produce, 1)
          aie.objectfifo.release @meta_34_33(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_35_34(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_34_33(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_34_33(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer13, %Buffer13, %UPDATEDHY13, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_35_34(Consume, 1)
        aie.objectfifo.release @act_34_33(Produce, 1)
        aie.objectfifo.release @meta_34_33(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_34_33(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY14, %Buffer14) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_34_33(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_34_33(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_34_33(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_33_32(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_33_32(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer14, %9, %UPDATEDHY14, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_34_33(Consume, 1)
          aie.objectfifo.release @meta_34_33(Consume, 1)
          aie.objectfifo.release @act_33_32(Produce, 1)
          aie.objectfifo.release @meta_33_32(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_34_33(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_33_32(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_33_32(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer14, %Buffer14, %UPDATEDHY14, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_34_33(Consume, 1)
        aie.objectfifo.release @act_33_32(Produce, 1)
        aie.objectfifo.release @meta_33_32(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @act_33_32(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        func.call @passThrough(%1, %UPDATEDHY15, %Buffer15) : (memref<1536xf32>, memref<512xf32>, memref<1536xf32>) -> ()
        aie.objectfifo.release @act_33_32(Consume, 1)
        %c0_0 = arith.constant 0 : index
        %c511 = arith.constant 511 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c511 step %c1_1 {
          %8 = aie.objectfifo.acquire @act_33_32(Consume, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %10 = aie.objectfifo.acquire @meta_33_32(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %12 = aie.objectfifo.acquire @act_out(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
          %14 = aie.objectfifo.acquire @meta_out(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
          %cst_2 = arith.constant 4.823000e-01 : f32
          %16 = arith.index_cast %arg1 : index to i32
          %c72_i32_3 = arith.constant 72 : i32
          %c512_i32_4 = arith.constant 512 : i32
          func.call @fdtd_2d(%Buffer15, %9, %UPDATEDHY15, %11, %13, %15, %cst_2, %c72_i32_3, %16, %c512_i32_4) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_33_32(Consume, 1)
          aie.objectfifo.release @meta_33_32(Consume, 1)
          aie.objectfifo.release @act_out(Produce, 1)
          aie.objectfifo.release @meta_out(Produce, 1)
        }
        %2 = aie.objectfifo.acquire @meta_33_32(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %4 = aie.objectfifo.acquire @act_out(Produce, 1) : !aie.objectfifosubview<memref<1536xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1536xf32>> -> memref<1536xf32>
        %6 = aie.objectfifo.acquire @meta_out(Produce, 1) : !aie.objectfifosubview<memref<1536xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
        %cst = arith.constant 4.823000e-01 : f32
        %c72_i32 = arith.constant 72 : i32
        %c511_i32 = arith.constant 511 : i32
        %c512_i32 = arith.constant 512 : i32
        func.call @fdtd_2d(%Buffer15, %Buffer15, %UPDATEDHY15, %3, %5, %7, %cst, %c72_i32, %c511_i32, %c512_i32) : (memref<1536xf32>, memref<1536xf32>, memref<512xf32>, memref<1536xui8>, memref<1536xf32>, memref<1536xui8>, f32, i32, i32, i32) -> ()
        aie.objectfifo.release @meta_33_32(Consume, 1)
        aie.objectfifo.release @act_out(Produce, 1)
        aie.objectfifo.release @meta_out(Produce, 1)
      }
      aie.end
    } {link_with = "fdtd_2d.o"}
    aiex.runtime_sequence(%arg0: memref<786432xf32>, %arg1: memref<786432xui8>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][64, 512, 3, 512][0, 1536, 512, 1]) {id = 1 : i64, metadata = @act_In} : memref<786432xf32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][64, 512, 3, 512][0, 1536, 512, 1]) {id = 0 : i64, metadata = @act_out} : memref<786432xf32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][64, 512, 3, 512][0, 1536, 512, 1]) {id = 2 : i64, metadata = @meta_mem_In} : memref<786432xui8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][64, 512, 3, 512][0, 1536, 512, 1]) {id = 2 : i64, metadata = @meta_mem_out} : memref<786432xui8>
      aiex.npu.sync {channel = 0 : i32, column = 3 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}

