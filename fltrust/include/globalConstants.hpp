#pragma once
#ifndef GLOBAL_CONSTANTS_HPP
#define GLOBAL_CONSTANTS_HPP

inline constexpr int BYZ_INDEX = 1;
inline constexpr int GLOBAL_ITERS = 5;
inline constexpr int GLOBAL_LEARN_RATE = 0.01;
inline constexpr uint64_t REG_SZ_DATA = 87360;
inline constexpr uint64_t MIN_SZ = 8;
inline constexpr size_t CAS_SIZE = sizeof(uint64_t);

// Flag codes
inline constexpr int SRVR_READ_WAIT = 0;    
inline constexpr int SRVR_READ_READY = 1;   // Server is ready to be read
inline constexpr int CLNT_READ_WAIT = 0;
inline constexpr int CLNT_READ_READY = 1;   // Client is ready to be read

// addr_locs of buffers
inline constexpr int SRVR_READY_IDX = 0;
inline constexpr int SRVR_W_IDX = 1;
inline constexpr int CLNT_READY_IDX = 2;
inline constexpr int CLNT_W_IDX = 3;

#endif // GLOBAL_CONSTANTS_HPP