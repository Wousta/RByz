#pragma once

inline constexpr int N_BYZ_CLNTS = 2;
inline constexpr int GLOBAL_ITERS = 100;
inline constexpr int CLNT_SUBSET_SIZE = 500;
inline constexpr int SRVR_SUBSET_SIZE = 500;
inline constexpr float GLOBAL_LEARN_RATE = 0.001;
inline constexpr uint64_t REG_SZ_DATA = 87360;
inline constexpr uint64_t MIN_SZ = 8;
inline constexpr size_t CAS_SIZE = sizeof(uint64_t);

// RBYZ
inline constexpr int GLOBAL_ITERS_RBYZ = 5;
inline constexpr uint64_t REG_SZ_CLNT = REG_SZ_DATA + 2 * sizeof(float);
constexpr int MEM_FREE = 0; 
constexpr int MEM_OCCUPIED = 1;


// addr_locs of buffers
inline constexpr int SRVR_READY_IDX = 0;
inline constexpr int SRVR_W_IDX = 1;
inline constexpr int CLNT_READY_IDX = 2;
inline constexpr int CLNT_W_IDX = 3;
inline constexpr int CLNT_CAS_IDX = 4;
