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
  RdmaOps &rdma_ops;
  IRegDatasetMngr &mngr;
  TrainInputParams t_params;
  std::vector<std::vector<int64_t>> step_times;

  void updateTS(std::vector<ClientDataRbyz> &clnt_data_vec,
                ClientDataRbyz &clnt_data, float srvr_loss,
                float srvr_error_rate);

  torch::Tensor
  aggregate_updates(const std::vector<torch::Tensor> &client_updates,
                    const torch::Tensor &w,
                    const std::vector<ClientDataRbyz> &clnt_data_vec,
                    const std::vector<uint32_t> &clnt_indices);

  void writeServerVD(RegMnistSplitter &splitter,
                     std::vector<ClientDataRbyz> &clnt_data);

  bool processVDOut(ClientDataRbyz &clnt_data, bool check_byz);
  void initTimeoutTime(std::vector<ClientDataRbyz> &clnt_data_vec);
  void runBenchMark(std::vector<ClientDataRbyz> &clnt_data_vec);
  void logTrustScores(const std::vector<ClientDataRbyz> &clnt_data_vec, int only_flt) const;

public:
  RByzAux(RdmaOps &rdma_ops, IRegDatasetMngr &mngr, TrainInputParams &t_params)
      : rdma_ops(rdma_ops), mngr(mngr), t_params(t_params),
        local_steps(t_params.local_steps_rbyz),
        global_rounds(t_params.global_iters_rbyz) {
          if (t_params.use_mnist) {
            step_times = {{2043}, {2143}, {2049}, {2144}, {2049}, {2148}, {2048}, {2148}, {2048}, {2143}};
          } else {
            step_times = {{32327}, {32325}, {32324}, {32326}, {32329}, {32322}, {32326}, {32325}, {32325}, {32325}};
          }
        }

  RByzAux() = delete;

  void awaitTermination(std::vector<ClientDataRbyz> &clnt_data_vec,
                        int rounds_rbyz);

  void runRByzClient(std::vector<torch::Tensor> &w, RegMemClnt &regMem);

  void runRByzServer(int n_clients, std::vector<torch::Tensor> &w,
                     RegMemSrvr &regMem,
                     std::vector<ClientDataRbyz> &clnt_data_vec);
};