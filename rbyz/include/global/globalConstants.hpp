#pragma once

inline constexpr int N_BYZ_CLNTS = 2;
inline constexpr int GLOBAL_ITERS = 100;
inline constexpr int DATASET_SIZE = 60000;
inline constexpr int CLNT_SUBSET_SIZE = 11500;
inline constexpr int SRVR_SUBSET_SIZE = 2500;
inline constexpr float GLOBAL_LEARN_RATE = 0.01f;
inline constexpr uint64_t REG_SZ_DATA = 87360;
inline constexpr uint64_t MIN_SZ = 8;
inline constexpr size_t CAS_SIZE = sizeof(uint64_t);

// RBYZ
inline constexpr int GLOBAL_ITERS_RBYZ = 3;
inline constexpr int LOCAL_STEPS_RBYZ = 5;
inline constexpr uint64_t REG_SZ_CLNT = REG_SZ_DATA + 2 * sizeof(float);
constexpr int MEM_FREE = 0; 
constexpr int MEM_OCCUPIED = 1;
constexpr int SRVR_READY_RBYZ = -1;


// addr_locs of buffers
inline constexpr int SRVR_READY_IDX = 0;
inline constexpr int SRVR_W_IDX = 1;
inline constexpr int CLNT_READY_IDX = 2;
inline constexpr int CLNT_W_IDX = 3;
inline constexpr int CLNT_LOSS_AND_ERR_IDX = 4;
inline constexpr int CLNT_CAS_IDX = 5;
inline constexpr int CLNT_LOCAL_STEP_IDX = 6;
inline constexpr int CLNT_IMAGES_IDX = 7;
inline constexpr int CLNT_LABELS_IDX = 8;
inline constexpr int CLNT_FORWARD_PASS_IDX = 9;
inline constexpr int CLNT_FORWARD_PASS_INDICES_IDX = 10;
