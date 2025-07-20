#pragma once
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include <atomic>
#include <cstdint>
#include <sys/types.h>
#include <vector>

/**
 * @brief Struct to hold the server's registered memory for FLtrust.
 * The server registers memory for each client and the server itself.
 */
struct RegMemSrvr {
  const int n_clients;
  const uint32_t reg_sz_data;     // Size of the registered parameter vector w
  uint32_t srvr_subset_size;
  uint32_t clnt_subset_size;
  uint32_t dataset_size;
  std::atomic<int> srvr_ready_flag;
  float *srvr_w;
  std::vector<int> clnt_ready_flags;
  std::atomic<int> srvr_ready_rb;
  std::vector<float *> clnt_ws;
  void *reg_data;

  RegMemSrvr(int n_clients, uint32_t reg_sz_data, void *reg_data)
      : reg_sz_data(reg_sz_data), n_clients(n_clients), srvr_ready_flag(0), srvr_ready_rb(0),
        clnt_ready_flags(n_clients, 0), clnt_ws(n_clients), reg_data(reg_data) {
          srvr_w = reinterpret_cast<float *>(malloc(reg_sz_data));
        }

  ~RegMemSrvr() {
    free(srvr_w);
  }
};

/**
 * @brief Holds the client's registered data.
 */
struct RegMemClnt {
  const int id;               
  const uint32_t reg_sz_data; // Size of the registered parameter vector w
  int srvr_ready_flag;
  float *srvr_w;
  float *clnt_w; 
  alignas(8) std::atomic<int> clnt_ready_flag;
  alignas(8) std::atomic<int> CAS;
  alignas(8) std::atomic<int> local_step;
  alignas(8) std::atomic<int> round;
  alignas(8) std::atomic<int> srvr_ready_rb;

  RegMemClnt(int id, int local_steps_rbyz, uint32_t reg_sz_data)
      : id(id), reg_sz_data(reg_sz_data), srvr_ready_flag(0), srvr_ready_rb(0),
        clnt_ready_flag(0), CAS(local_steps_rbyz), local_step(0), round(1) {
    srvr_w = reinterpret_cast<float *>(malloc(reg_sz_data));
    clnt_w = reinterpret_cast<float *>(malloc(reg_sz_data));
  }

  ~RegMemClnt() {
    free(srvr_w);
    free(clnt_w);
  }
};

/**
 * @brief Struct to hold the client's registered data for RByz.
 * Used by the server to read the client's data.
 */
struct ClientDataRbyz {
  int index;
  bool is_byzantine = false;
  bool include_in_agg = false;
  float trust_score = 0.0;
  float *updates;

  // Handling of slow clients
  bool is_slow = false;
  int next_step = 1;
  int max_step = 0;
  int min_step = 0;
  int steps_to_finish = 0;
  std::chrono::milliseconds limit_step_time;

  // VD out data, average errors and losses of the forward pass of the samples inserted by the server to this client
  // The server will have a different loss and error rate of the samples that it inserted to the client
  alignas(8) float loss = 777.0f;         // These are high initial values to not mess up the UpdateTrustScore function
  alignas(8) float error_rate = 777.0f;   // That needs to find the minimum loss and error rate of all clients
  alignas(8) float loss_srvr;             // These are initialized before checking this client by RByzAux::processVDOut
  alignas(8) float error_rate_srvr;

  // Dataset data
  alignas(8) size_t dataset_size;
  uint64_t num_samples;
  std::unordered_set<size_t> inserted_indices;

  // For RByz
  alignas(8) std::atomic<int> clnt_CAS; // Used for client's local steps to finish
  alignas(8) int local_step = 0;
  alignas(8) int round = 0;

  // Forward pass data
  size_t forward_pass_mem_size;
  size_t forward_pass_indices_mem_size;
  float *forward_pass;
  uint32_t *forward_pass_indices;

  ClientDataRbyz()
      : clnt_CAS(0), trust_score(0) {}

  ~ClientDataRbyz() {
    // Only free memory that was allocated with malloc/new
    free(updates);
    free(forward_pass_indices);
  }

  void init(int local_steps_rbyz) {
    clnt_CAS.store(local_steps_rbyz);
    max_step = local_steps_rbyz;
    min_step = local_steps_rbyz;
    steps_to_finish = local_steps_rbyz;
  }
};

struct TrainInputParams {
  // Logging
  std::string logs_dir = "";
  std::string ts_file = "";
  std::string acc_file= "";
  std::string included_agg_file = "";
  std::string clnts_renew_file = "clnts_used_to_renew.log";

  bool use_mnist = false;
  int n_clients;
  int n_byz_clnts;
  int epochs;
  int batch_size;
  float global_learn_rate;
  double local_learn_rate;
  int clnt_subset_size;
  int srvr_subset_size;
  int global_iters_fl;

  // RByz specific parameters
  int local_steps_rbyz;
  int global_iters_rbyz;
  int chunk_size;
  float clnt_vd_proportion;   // Proportion of validation data for each client (proportion of total chunks writable on client)
  float vd_prop_write;        // Proportion of total chunks writable on client to write each time the test is renewed
  int test_renewal_freq;      // Frequency of test renewal (every n rounds)
  int overwrite_poisoned; // Allow VD samples to overwrite poisoned samples
  int wait_all; // Ignore slow clients in the trust score calculation
  float batches_fpass_prop = 0.0;

  //misc
  int only_flt;
  int label_flip_type;
  float flip_ratio;
  int srvr_wait_inc = 0;    // Server wait increment for slow clients in timeouts experiment
};