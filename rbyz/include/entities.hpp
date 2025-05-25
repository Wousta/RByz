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
  int srvr_ready_flag;
  float* srvr_w;
  int clnt_ready_flag;
  float* clnt_w;
  float* loss_and_err;
  std::atomic<int> clnt_CAS;
  std::atomic<int> local_step;
  std::atomic<int> round;

  RegMemClnt() : srvr_ready_flag(0), clnt_ready_flag(0), clnt_CAS(MEM_FREE), local_step(0), round(0) {
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
    bool is_byzantine = false;
    std::atomic<int> clnt_CAS;
    float trust_score;
    float* updates;
    float* loss;       
    float* error_rate;  
    int local_step = 0;
    int round = 0;

    // Dataset data used
    size_t dataset_size;  // Size of the registered dataset
    std::vector<size_t> inserted_indices;  // Indices the server put a test into that might be in the forward pass table

    // Forward pass data used
    size_t forward_pass_mem_size;
    size_t forward_pass_indices_mem_size;
    float* forward_pass;
    uint32_t* forward_pass_indices;

    ~ClientDataRbyz() {
        // Only free memory that was allocated with malloc/new
        if (updates) free(updates);
        if (loss) free(loss);
        if (error_rate) free(error_rate);
        if (forward_pass_indices) free(forward_pass_indices);
    }
};