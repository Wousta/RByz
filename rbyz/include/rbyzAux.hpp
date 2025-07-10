#pragma once

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "datasetLogic/regMnistSplitter.hpp"
#include "entities.hpp"
#include "rdmaOps.hpp"
#include <float.h>
#include <vector>

class RByzAux {
private:
  const int local_steps;
  const int global_rounds;
  const bool byz_clnt;
  float ts_threshold;
  RdmaOps &rdma_ops;
  IRegDatasetMngr &mngr;
  TrainInputParams t_params;
  std::vector<std::vector<int64_t>> step_times;
  std::mt19937 rng;
  std::bernoulli_distribution coin_flip{0.5};  // 50% chance

  // Timeout params
  int min_steps;      // Minimum steps to consider a client valid
  int middle_steps;   
  std::uniform_int_distribution<int> step_range;

  void updateTS(std::vector<ClientDataRbyz> &clnt_data_vec,
                ClientDataRbyz &clnt_data, float srvr_loss,
                float srvr_error_rate);

  torch::Tensor
  aggregate_updates(const std::vector<torch::Tensor> &client_updates,
                    const std::vector<ClientDataRbyz> &clnt_data_vec,
                    const std::vector<uint32_t> &clnt_indices);

  void writeServerVD(RegMnistSplitter &splitter,
                     std::vector<ClientDataRbyz> &clnt_data, float proportion);

  bool processVDOut(ClientDataRbyz &clnt_data, bool check_byz);
  void initTimeoutTime(std::vector<ClientDataRbyz> &clnt_data_vec);
  void runBenchMark(std::vector<ClientDataRbyz> &clnt_data_vec);
  void logTrustScores(const std::vector<ClientDataRbyz> &clnt_data_vec, int only_flt) const;
  void waitTimeout(ClientDataRbyz& clnt_data, int round);
  void waitInfinite(ClientDataRbyz& clnt_data, int round);

public:
  void *extra_vd_col = nullptr;
  uint32_t extra_vd_col_sz = 0;
  uint32_t extra_vd_col_max_samples = 0;

  RByzAux(RdmaOps &rdma_ops, IRegDatasetMngr &mngr, TrainInputParams &t_params)
      : rdma_ops(rdma_ops), mngr(mngr), t_params(t_params),
        local_steps(t_params.local_steps_rbyz),
        global_rounds(t_params.global_iters_rbyz),
        byz_clnt(mngr.worker_id <= t_params.n_byz_clnts),
        rng(std::random_device{}()) {

          min_steps = std::ceil(local_steps * 0.5);
          middle_steps = std::ceil(local_steps * 0.75);
          step_range = std::uniform_int_distribution<int>(middle_steps, local_steps);

          if (t_params.use_mnist) {
            ts_threshold = 0.91; // Benchmark threshold
            if (t_params.n_clients == 10) {
              step_times = {{2043}, {2143}, {2049}, {2144}, {2049}, {2148}, {2048}, {2148}, {2048}, {2143}};
            } else {
              for (int i = 0; i < t_params.n_clients; i++) {
                step_times.push_back({2144});
              }
            }
          } else {
            ts_threshold = 0.87; // Benchmark threshold
            if (t_params.n_clients == 10) {
              step_times = {{29389}, {29387}, {29386}, {29388}, {29390}, {29384}, {29388}, {29387}, {29387}, {29387}};
            } else {
              for (int i = 0; i < t_params.n_clients; i++) {
                step_times.push_back({29389}); 
              }
            }
          }
        }

  RByzAux() = delete;

  void awaitTermination(std::vector<ClientDataRbyz> &clnt_data_vec,
                        int rounds_rbyz);

  void runRByzClient(std::vector<torch::Tensor> &w, RegMemClnt &regMem);

  void runRByzServer(int n_clients, std::vector<torch::Tensor> &w,
                     RegMemSrvr &regMem,
                     std::vector<ClientDataRbyz> &clnt_data_vec);

  inline bool coinFlip() { return coin_flip(rng); }
  
};