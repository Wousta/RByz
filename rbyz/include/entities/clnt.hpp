#pragma once
#include <vector>
#include <atomic>

#include "global/globalConstants.hpp"

/**
 * @brief Holds the client's registered data.
 */
struct RegMemClnt {
  int srvr_ready_flag;
  float* srvr_w;
  int clnt_ready_flag;
  float* clnt_w;
  float* loss_and_err;
  std::atomic<int> clnt_CAS;
  std::atomic<int> local_step;

  RegMemClnt() : srvr_ready_flag(0), clnt_ready_flag(0), clnt_CAS(MEM_OCCUPIED), local_step(0) {
    srvr_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
    clnt_w = reinterpret_cast<float*> (malloc(REG_SZ_CLNT));
    loss_and_err = reinterpret_cast<float*> (malloc(MIN_SZ));
  }


  ~RegMemClnt() {
    free(srvr_w);
    free(clnt_w);
    free(loss_and_err);
  }
};

void registerClntMemory(RegInfo& reg_info, RegMemClnt& regMem, RegisteredMnistTrain& mnist);
