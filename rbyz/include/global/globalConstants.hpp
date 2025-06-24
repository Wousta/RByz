#pragma once

inline constexpr int N_BYZ_CLNTS = 3;
inline constexpr int GLOBAL_ITERS_FL = 4;
inline constexpr float GLOBAL_LEARN_RATE = 0.05f;
inline constexpr uint64_t MIN_SZ = 8;
inline constexpr double ACCURACY_WARMUP = 0.5;
inline constexpr double GLOBAL_TARGET_ACCURACY = 0.96;
inline constexpr size_t CAS_SIZE = sizeof(uint64_t);
inline constexpr int FLTRUST_END = -3;  
inline constexpr int SGL_SIZE = 2;  
inline constexpr bool MNIST = true;  
inline constexpr bool CIFAR10 = false;  

// MNIST
inline constexpr uint64_t REG_SZ_DATA_MNIST = 87360;
inline constexpr int DATASET_SIZE_MNIST = 60000;
inline constexpr int CLNT_SUBSET_SIZE_MNIST = 5900;   
inline constexpr int SRVR_SUBSET_SIZE_MNIST = 1000;  

// CIFAR10
inline constexpr uint64_t REG_SZ_DATA_CF10 = 248024;
inline constexpr int DATASET_SIZE_CF10 = 50000;
inline constexpr int CLNT_SUBSET_SIZE_CF10 = 24500;
inline constexpr int SRVR_SUBSET_SIZE_CF10 = 1000;

// RBYZ
inline constexpr int GLOBAL_ITERS_RBYZ = 10;
inline constexpr int LOCAL_STEPS_RBYZ = 5;
inline constexpr int VD_SPLIT = 100;  // will divide the VD per client, to determine how many samples to use per VD for each client.
constexpr int MEM_FREE = 0; 
constexpr int MEM_OCCUPIED = 1;
constexpr int SRVR_READY_RBYZ = -1;
constexpr int SRVR_FINISHED_RBYZ = -2;


// addr_locs of buffers
inline constexpr int SRVR_READY_IDX = 0;
inline constexpr int SRVR_W_IDX = 1;
inline constexpr int CLNT_READY_IDX = 2;
inline constexpr int CLNT_W_IDX = 3;
inline constexpr int CLNT_LOSS_AND_ERR_IDX = 4;
inline constexpr int CLNT_CAS_IDX = 5;
inline constexpr int CLNT_LOCAL_STEP_IDX = 6;
inline constexpr int CLNT_ROUND_IDX = 7;
inline constexpr int REG_DATASET_IDX = 8;
inline constexpr int CLNT_FORWARD_PASS_IDX = 9;
inline constexpr int CLNT_FORWARD_PASS_INDICES_IDX = 10;
