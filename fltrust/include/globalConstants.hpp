#pragma once

inline constexpr int N_BYZ_CLNTS = 3;
inline constexpr int GLOBAL_ITERS = 56;
inline constexpr int DATASET_SIZE = 60000;
inline constexpr int CLNT_SUBSET_SIZE = 5900;   
inline constexpr int SRVR_SUBSET_SIZE = 1000;  
inline constexpr float GLOBAL_LEARN_RATE = 0.01f;
inline constexpr uint64_t REG_SZ_DATA = 87360;
inline constexpr uint64_t MIN_SZ = 8;
inline constexpr size_t CAS_SIZE = sizeof(uint64_t);

// addr_locs of buffers
inline constexpr int SRVR_READY_IDX = 0;
inline constexpr int SRVR_W_IDX = 1;
inline constexpr int CLNT_READY_IDX = 2;
inline constexpr int CLNT_W_IDX = 3;
