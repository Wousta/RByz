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
                    const std::vector<ClientDataRbyz> &clnt_data_vec,
                    const std::vector<uint32_t> &clnt_indices);

  void writeServerVD(RegMnistSplitter &splitter,
                     std::vector<ClientDataRbyz> &clnt_data, float proportion);

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
            if (t_params.n_clients == 10) {
              step_times = {{2043}, {2143}, {2049}, {2144}, {2049}, {2148}, {2048}, {2148}, {2048}, {2143}};
            } else {
              for (int i = 0; i < t_params.n_clients; i++) {
                step_times.push_back({2144});
              }
            }
          } else {
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
};