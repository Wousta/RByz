#pragma once
#include "datasetLogic/iRegDatasetMngr.hpp"
#include "global/globalConstants.hpp"
#include <atomic>
#include <vector>

/**
 * @brief Struct to hold the server's registered memory for FLtrust.
 * The server registers memory for each client and the server itself.
 */
struct RegMemSrvr {
  const int n_clients;
  const uint32_t reg_sz_data;
  uint32_t srvr_subset_size;
  uint32_t clnt_subset_size;
  uint32_t dataset_size;
  int srvr_ready_flag = 0;
  float *srvr_w;
  std::vector<int> clnt_ready_flags;
  std::vector<float *> clnt_ws;
  std::vector<float *> clnt_loss_and_err;
  void* reg_data;

  RegMemSrvr(int n_clients, uint32_t reg_sz_data, IRegDatasetMngr &manager)
      : srvr_w(reinterpret_cast<float *>(malloc(reg_sz_data))),
        reg_sz_data(reg_sz_data),
        n_clients(n_clients),
        clnt_ready_flags(n_clients, 0),
        clnt_ws(n_clients),
        clnt_loss_and_err(n_clients),
        reg_data(manager.data_info.reg_data) {}

  ~RegMemSrvr() {
    free(srvr_w);
    for (int i = 0; i < n_clients; i++) {
      free(clnt_ws[i]);
      free(clnt_loss_and_err[i]);
    }
  }
};

/**
 * @brief Holds the client's registered data.
 */
struct RegMemClnt {
  const int id; // For identification purposes, not used in RByz
  const uint32_t reg_sz_data; // Size of the registered parameter vector w
  int srvr_ready_flag;
  float* srvr_w;
  int clnt_ready_flag;
  float* clnt_w;
  float* loss_and_err;
  alignas(8) std::atomic<int> CAS;
  alignas(8) std::atomic<int> local_step;
  alignas(8) std::atomic<int> round;

  RegMemClnt(int id, uint32_t reg_sz_data) : 
      id(id), reg_sz_data(reg_sz_data), srvr_ready_flag(0), 
      clnt_ready_flag(0), CAS(LOCAL_STEPS_RBYZ), 
      local_step(0), round(0) {
    srvr_w = reinterpret_cast<float*> (malloc(reg_sz_data));
    clnt_w = reinterpret_cast<float*> (malloc(reg_sz_data));
    loss_and_err = reinterpret_cast<float*> (malloc(MIN_SZ));
  }

  ~RegMemClnt() {
    free(srvr_w);
    free(clnt_w);
    free(loss_and_err);
  }
};

/**
 * @brief Struct to hold the client's registered data for RByz.
 * Used by the server to read the client's data.
 */
struct ClientDataRbyz {
    int index;
    bool is_byzantine = false;
    float trust_score;
    float* updates;
    float* loss;        // Unused in RByz, but kept for compatibility
    float* error_rate;  // Unused in RByz, but kept for compatibility

    // Handling of slow clients
    bool is_slow = false;
    int next_step = 1; 
    int max_step = LOCAL_STEPS_RBYZ;
    int min_step = LOCAL_STEPS_RBYZ;
    int steps_to_finish = LOCAL_STEPS_RBYZ;
    std::chrono::milliseconds limit_step_time;
    
    // Dataset data
    alignas(8) size_t dataset_size;
    std::unordered_set<size_t> inserted_indices;

    // Forward pass data
    size_t forward_pass_mem_size;
    size_t forward_pass_indices_mem_size;
    float* forward_pass;
    uint32_t* forward_pass_indices;

    // VD out data
    alignas(8) float loss_clnt;
    alignas(8) float error_rate_clnt;
    alignas(8) float loss_srvr;
    alignas(8) float error_rate_srvr;
    
    alignas(8) std::atomic<int> clnt_CAS;    
    alignas(8) int local_step = 0;           
    alignas(8) int round = 0;               

    ClientDataRbyz() : clnt_CAS(LOCAL_STEPS_RBYZ), trust_score(0) {}

    ~ClientDataRbyz() {
        // Only free memory that was allocated with malloc/new
        if (updates) free(updates);
        if (loss) free(loss);
        if (error_rate) free(error_rate);
        if (forward_pass_indices) free(forward_pass_indices);
    }
};