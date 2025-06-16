#pragma once
#include <atomic>
#include <vector>

/**
 * @brief Struct to hold the server's registered memory for FLtrust.
 * The server registers memory for each client and the server itself.
 */
struct RegMemSrvr {
private:
  int n_clients;

public:
  int srvr_ready_flag = 0;
  float *srvr_w;
  std::vector<int> clnt_ready_flags;
  std::vector<float *> clnt_ws;
  std::vector<float *> clnt_loss_and_err;
  void* vd_sample;

  RegMemSrvr(int n_clients, size_t sample_size)
      : vd_sample(reinterpret_cast<void *>(malloc(sample_size))),
        srvr_w(reinterpret_cast<float *>(malloc(REG_SZ_DATA))),
        n_clients(n_clients),
        clnt_ready_flags(n_clients, 0),
        clnt_ws(n_clients),
        clnt_loss_and_err(n_clients) {}

  ~RegMemSrvr() {
    free(srvr_w);
    for (int i = 0; i < n_clients; i++) {
      free(clnt_ws[i]);
      free(clnt_loss_and_err[i]);
      free(vd_sample);
    }
  }
};

/**
 * @brief Holds the client's registered data.
 */
struct RegMemClnt {
  const int id; // For identification purposes, not used in RByz
  int srvr_ready_flag;
  float* srvr_w;
  int clnt_ready_flag;
  float* clnt_w;
  float* loss_and_err;
  alignas(8) std::atomic<int> clnt_CAS;
  alignas(8) std::atomic<int> local_step;
  alignas(8) std::atomic<int> round;

  RegMemClnt(int id) : 
      id(id), srvr_ready_flag(0), 
      clnt_ready_flag(0), clnt_CAS(MEM_FREE), 
      local_step(0), round(0) {
    srvr_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
    clnt_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
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
    int timeouts = 0;
    bool is_byzantine = false;
    float trust_score;
    float* updates;
    float* loss;        // Unused in RByz, but kept for compatibility
    float* error_rate;  // Unused in RByz, but kept for compatibility
    
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

    ClientDataRbyz() : clnt_CAS(MEM_FREE), trust_score(0) {}

    ~ClientDataRbyz() {
        // Only free memory that was allocated with malloc/new
        if (updates) free(updates);
        if (loss) free(loss);
        if (error_rate) free(error_rate);
        if (forward_pass_indices) free(forward_pass_indices);
    }
};