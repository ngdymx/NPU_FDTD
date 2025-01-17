#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../../../../mlir-aie/runtime_lib/test_lib/test_utils.h"

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

    // ------------------------------------------------------
    // Parse program arguments
    // ------------------------------------------------------
    po::options_description desc("Allowed options");
    po::variables_map vm;
    test_utils::add_default_options(desc);

    test_utils::parse_options(argc, argv, desc, vm);
    int verbosity = vm["verbosity"].as<int>();
    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();
    int trace_size = vm["trace_sz"].as<int>();

    constexpr int COLS = 256;
    constexpr int ROWS = 256;
    constexpr int BLOCK_SIZE = 16;
    constexpr int CHANNELIN0_SIZE = COLS * ROWS * 3;
    constexpr int CHANNELIN1_SIZE = COLS * ROWS * 4;
    constexpr int CHANNELOUT0_SIZE = COLS * ROWS * 3;
    constexpr int CHANNELOUT1_SIZE = COLS * ROWS * 4;

    constexpr int T = 4 * 64;

    constexpr int SRC_POINT_ROW = 120;
    constexpr int SRC_POINT_COL = 120;

    float Ez[ROWS][COLS];
    float Hx[ROWS][COLS];
    float Hy[ROWS][COLS];
    uint8_t C_index[ROWS][COLS];
    uint8_t Ca_index[ROWS][COLS];
    uint8_t SRC_index[ROWS][COLS];
    uint8_t Mask_in[ROWS][COLS];

    // float channel_in0[CHANNELIN0_SIZE];
    float channel_in0[ROWS][COLS / BLOCK_SIZE][3][BLOCK_SIZE];
    uint8_t channel_in1[ROWS][COLS / BLOCK_SIZE][4][BLOCK_SIZE];

    for (int i = 0; i < ROWS; i++){
        for (int j = 0; j < COLS; j++){
            SRC_index[i][j] = 0; 
            Ez[i][j] = 0; 
            Hx[i][j] = 0; 
            Hy[i][j] = 0; 
        }
    }
      
    FILE *fp;
    uint8_t temp_c[ROWS][COLS];
    uint8_t temp_ca[ROWS][COLS];
    fp = fopen("./host/C.bin", "r");

    if (fp == NULL){
        printf("File not found!\n");
        return -1;
    }
    fread((uint8_t*)(&temp_c), sizeof(uint8_t), ROWS*COLS, fp);
    fclose(fp);
    for (int i = 0; i < ROWS; i++){
        for (int j = 0; j < COLS; j++){
            C_index[i][j] = temp_c[i][j];
        }
    }

    *fp;
    fp = fopen("./host/Ca.bin", "r");

    if (fp == NULL){
        printf("File not found!\n");
        return -1;
    }
    fread((uint8_t*)(&temp_ca), sizeof(uint8_t), ROWS*COLS, fp);
    fclose(fp);
    for (int i = 0; i < ROWS; i++){
        for (int j = 0; j < COLS; j++){
            Ca_index[i][j] = temp_ca[i][j];
        }
    }

    for (int i = 0; i < ROWS; i++){
        for (int j = 0; j < COLS; j++){
            if((i == SRC_POINT_ROW) && (j == SRC_POINT_COL)){
                Mask_in[i][j] = 1;
            }
            else{
                Mask_in[i][j] = 0;
            }
        }
    }

    for (int row = 0; row < ROWS; row++){
        for (int block = 0; block < COLS / BLOCK_SIZE; block++){
            for (int i = 0; i < BLOCK_SIZE; i++){
                channel_in0[row][block][0][i] = Ez[row][i + block * BLOCK_SIZE];
                channel_in0[row][block][1][i] = Hx[row][i + block * BLOCK_SIZE];
                channel_in0[row][block][2][i] = Hy[row][i + block * BLOCK_SIZE];
                channel_in1[row][block][0][i] = C_index[row][i + block * BLOCK_SIZE];
                channel_in1[row][block][1][i] = Ca_index[row][i + block * BLOCK_SIZE];
                channel_in1[row][block][2][i] = SRC_index[row][i + block * BLOCK_SIZE];
                channel_in1[row][block][3][i] = Mask_in[row][i + block * BLOCK_SIZE];
            }
        }
    }

    // Load instruction sequence
    std::vector<uint32_t> instr_v = test_utils::load_instr_sequence(vm["instr"].as<std::string>());
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    // ------------------------------------------------------
    // Get device, load the xclbin & kernel and register them
    // ------------------------------------------------------
    // Get a device handle
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    // Load the xclbin
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

    // Load the kernel
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
    std::string Node = vm["kernel"].as<std::string>();

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),[Node, verbosity](xrt::xclbin::kernel &k) {
        auto name = k.get_name();
        if (verbosity >= 1) {
            std::cout << "Name: " << name << std::endl;
        }
        return name.rfind(Node, 0) == 0;
    });
    auto kernelName = xkernel.get_name();

    // Register xclbin
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()<< "\n";
    device.register_xclbin(xclbin);

    // Get a hardware context
    std::cout << "Getting hardware context.\n";
    xrt::hw_context context(device, xclbin.get_uuid());

    // Get a kernel handle
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
    auto kernel = xrt::kernel(context, kernelName);

    // ------------------------------------------------------
    // Initialize input/ output buffer sizes and sync them
    // ------------------------------------------------------

    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inout = xrt::bo(device, CHANNELIN0_SIZE * sizeof(float), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_meta = xrt::bo(device, CHANNELIN1_SIZE * sizeof(uint8_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    std::cout << "Writing data into buffer objects.\n";

    float *bufInOut = bo_inout.map<float *>();
    memcpy(bufInOut, channel_in0, (CHANNELIN0_SIZE * sizeof(float)));

    uint8_t *bufmeta = bo_meta.map<uint8_t *>();
    memcpy(bufmeta, channel_in1, (CHANNELIN1_SIZE * sizeof(uint8_t)));

    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_meta.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < T; it++){
        std::cout << "Running Kernel.\n"; 
        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inout, bo_meta); 
        std::cout << "Finishing Kernel.\n"; 
        run.wait();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << std::endl << "NPU time: " << time << "us." << std::endl;
    float macs = ROWS * COLS * 4 + ROWS * COLS * 4 + ROWS * COLS * 7 ;
    std::cout << "Avg NPU gflops: " << macs / (1000 * time / T) << std::endl;

    bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    float *Out = bo_inout.map<float *>();
    float Out_Ez[ROWS][COLS];
    float Out_Hx[ROWS][COLS];
    float Out_Hy[ROWS][COLS];

    uint8_t *Out_meta = bo_meta.map<uint8_t *>();
    uint8_t out_meta[ROWS][COLS];

    for (int row = 0; row < ROWS; row++){
        for (int col = 0; col < COLS; col++){
                Out_Ez[row][col] = Out[row * COLS * 3 + (col / 16) * 48 + col % 16];
                Out_Hx[row][col] = Out[row * COLS * 3 + (col / 16) * 48 + 16 + col % 16];
                Out_Hy[row][col] = Out[row * COLS * 3 + (col / 16) * 48 + 2*16 + col % 16];
        }
    }

//    printf("Ez =\n");
//    for(int row = 110; row < 130; row++){
//        for (int i = 110; i < 130; i++){
//            printf("%8.3f ", Out_Ez[row][i]);
//        }
//        printf("\n");
//    }
    *fp;
    fp = fopen("./u.bin", "w");
    fwrite((float*)(&Out_Ez), sizeof(float), ROWS*COLS, fp);
    fclose(fp);

    printf("\n");
    printf("src_index =\n");
    for (int row = 0; row < ROWS; row++){
        for (int col = 0; col < COLS; col++){
                out_meta[row][col] = Out_meta[row * COLS * 3 + (col / 16) * 48 + 32 + col % 16];
        }
    }
    printf("%d\n ", out_meta[0][0]);
}
