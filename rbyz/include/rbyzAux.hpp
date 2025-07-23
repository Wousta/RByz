#pragma once

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "datasetLogic/regMnistSplitter.hpp"
#include "entities.hpp"
#include "logger.hpp"
#include "rdmaOps.hpp"
#include <float.h>
#include <random>
#include <vector>

#define TIMEOUT_MULT_ 1.3
#define EXP_BACK_MULT_ 1.5
#define TIMEOUT_MNIST 696
#define TIMEOUT_CIFAR10 3986

using millis = std::chrono::milliseconds;

class RByzAux {
private:
  const float TIMEOUT_SLACK = 1.4;
  const float STEP_TIME_REDUCE = 0.95;
  const int local_steps;
  const int global_rounds;
  const bool byz_clnt;
  float ts_threshold; // Trust score threshold for VD extra column client
                      // selection (not yet used)
  RdmaOps &rdma_ops;
  IRegDatasetMngr &mngr;
  TrainInputParams t_params;
  std::vector<std::vector<int>> step_times;
  std::mt19937 rng;
  std::bernoulli_distribution coin_flip{0.5}; // 50% chance

  struct acc_info {
    float threshold_acc;    // Accuracy threshold for convergence
    int rounds_to_converge; // Number of rounds it took rbyz to reach a
                            // threshold accuracy
  } acc_info_;

  struct timeouts {
    int min_steps; // Minimum steps to consider a client valid
    int mid_steps; // Middle steps for the step range distribution lower bound
    std::uniform_int_distribution<int> step_range;
  } timeouts_;

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
  void logTrustScores(const std::vector<ClientDataRbyz> &clnt_data_vec,
                      int only_flt) const;
  void waitTimeout(ClientDataRbyz &clnt_data, int round);
  void waitInfinite(ClientDataRbyz &clnt_data, int round);
  void renewTrustedClientsColumn(RegMnistSplitter &splitter,
                                 std::vector<ClientDataRbyz> &clnt_data_vec);

  inline void runSteps(std::vector<ClientDataRbyz> &clnt_data_vec, int round) {
    for (ClientDataRbyz &clnt_data : clnt_data_vec) {
      rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX,
                              clnt_data.index);
      rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, clnt_data.index);

      if (clnt_data.is_byzantine) {
        Logger::instance().log("  WARNING: skipping Client " +
                                std::to_string(clnt_data.index) + "\n");
        continue;
      }

      if (t_params.wait_all) {
        waitInfinite(clnt_data, round);
      } else {
        waitTimeout(clnt_data, round);
      }
    }
  }

  inline int waitExpBackoffStep(millis &exp_backoff_time) {
    std::this_thread::sleep_for(exp_backoff_time);
    long new_backoff_time =
        static_cast<long>(exp_backoff_time.count() * EXP_BACK_MULT_);
    exp_backoff_time = millis(new_backoff_time);

    return exp_backoff_time.count();
  }

  inline int waitInitialTime(ClientDataRbyz &clnt_data, int round) {
    millis initial_time(static_cast<long>(clnt_data.limit_step_time.count()));

    int time = 0;
    if (clnt_data.local_step < clnt_data.next_step &&
        clnt_data.round <= round) {
      std::this_thread::sleep_for(initial_time);
      time = initial_time.count();
      Logger::instance().log("     -> Server waited initial: " +
                             std::to_string(initial_time.count()) +
                             " ms for client " +
                             std::to_string(clnt_data.index) + "\n");
    }

    return time;
  }

public:
  void *extra_vd_col = nullptr;
  uint32_t extra_vd_col_sz = 0;
  uint32_t extra_vd_col_max_samples = 0;

  RByzAux(RdmaOps &rdma_ops, IRegDatasetMngr &mngr, TrainInputParams &t_params)
      : rdma_ops(rdma_ops), mngr(mngr), t_params(t_params),
        local_steps(t_params.local_steps_rbyz),
        global_rounds(t_params.global_iters_rbyz),
        byz_clnt(mngr.worker_id <= t_params.n_byz_clnts), rng(14) {

    timeouts_.min_steps = std::floor(local_steps * 0.5);
    timeouts_.mid_steps = std::ceil(local_steps * 0.75);
    timeouts_.step_range =
        std::uniform_int_distribution<int>(timeouts_.mid_steps, local_steps);
    acc_info_.rounds_to_converge = global_rounds;
    acc_info_.threshold_acc = t_params.use_mnist ? 96.0 : 81.0;
    ts_threshold = t_params.use_mnist ? 0.92 : 0.65; // Benchmark threshold

    int timeout = t_params.use_mnist ? TIMEOUT_MNIST : TIMEOUT_CIFAR10;
    for (int i = 0; i < t_params.n_clients; i++) {
      step_times.push_back({timeout});
    }
  }

  RByzAux() = delete;

  void awaitTermination(std::vector<ClientDataRbyz> &clnt_data_vec, int code);
  void runRByzClient(std::vector<torch::Tensor> &w, RegMemClnt &regMem);
  void runRByzServer(int n_clients, std::vector<torch::Tensor> &w,
                     RegMemSrvr &regMem,
                     std::vector<ClientDataRbyz> &clnt_data_vec);
  inline bool coinFlip() { return coin_flip(rng); }
  int getRoundsToConverge() const { return acc_info_.rounds_to_converge; }
};