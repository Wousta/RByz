#pragma once

#include <float.h>

#include <random>
#include <vector>

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "datasetLogic/regMnistSplitter.hpp"
#include "entities.hpp"
#include "logger.hpp"
#include "rdmaOps.hpp"

using millis = std::chrono::milliseconds;

class RByzClnt {
 private:
  const float TIMEOUT_SLACK = 1.4;
  const float STEP_TIME_REDUCE = 0.95;
  const int local_steps;
  const int global_rounds;
  const bool byz_clnt;

  RdmaOps &rdma_ops;
  IRegDatasetMngr &mngr;
  TrainInputParams t_params;
  std::mt19937 rng;
  std::bernoulli_distribution coin_flip{0.5};  // 50% chance

  inline bool coinFlip() { return coin_flip(rng); }

 public:
  RByzClnt(RdmaOps &rdma_ops, IRegDatasetMngr &mngr, TrainInputParams &t_params)
      : rdma_ops(rdma_ops),
        mngr(mngr),
        t_params(t_params),
        local_steps(t_params.local_steps_rbyz),
        global_rounds(t_params.global_iters_rbyz),
        byz_clnt(mngr.worker_id <= t_params.n_byz_clnts),
        rng(14) {}

  RByzClnt() = delete;

  void runRByzClient(std::vector<torch::Tensor> &w, RegMemClnt &regMem);
};