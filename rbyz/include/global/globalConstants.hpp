#pragma once

inline constexpr uint64_t NUM_CLASSES = 10; // Number of classes of CIFAR-10 and MNIST
inline constexpr uint64_t MIN_SZ = 8;
inline constexpr size_t CAS_SIZE = sizeof(uint64_t);
inline constexpr int SRVR_FINISHED = -5;  

// MNIST
inline constexpr uint64_t REG_SZ_DATA_MNIST = 87360;    // Size of the parameter vector w for MNIST
inline constexpr int DATASET_SIZE_MNIST = 60000;

// CIFAR10
inline constexpr int DATASET_SIZE_CF10 = 50000;

// RBYZ
inline constexpr int MEM_FREE = 0; 
inline constexpr int MEM_OCCUPIED = 1;
inline constexpr int SRVR_READY_RBYZ = -1;
inline constexpr int SRVR_FINISHED_RBYZ = -2;
inline constexpr int CLNT_FINISHED_RBYZ = -3;

// Byz attack settings
inline constexpr int NO_ATTACK = 0;
inline constexpr int RANDOM_FLIP = 1;
inline constexpr int CORRUPT_IMAGES_RNG = 2;
inline constexpr int TARGETED_FLIP_1 = 3;
inline constexpr int TARGETED_FLIP_2 = 4;
inline constexpr int TARGETED_FLIP_3 = 5;
inline constexpr int TARGETED_FLIP_4 = 6;

// addr_locs of buffers
inline constexpr int SRVR_W_IDX = 0;
inline constexpr int CLNT_W_IDX = 1;
inline constexpr int SRVR_READY_IDX = 2;
inline constexpr int CLNT_READY_IDX = 3;
inline constexpr int SRVR_READY_RB_IDX = 4;
inline constexpr int CLNT_CAS_IDX = 5;
inline constexpr int CLNT_LOCAL_STEP_IDX = 6;
inline constexpr int CLNT_ROUND_IDX = 7;
inline constexpr int REG_DATASET_IDX = 8;
inline constexpr int CLNT_FORWARD_PASS_IDX = 9;
inline constexpr int CLNT_FORWARD_PASS_INDICES_IDX = 10;
