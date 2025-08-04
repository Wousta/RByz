#include "entities.hpp"
#include "manager/iRegDatasetMngr.hpp"
#include "rdmaOps.hpp"

class FLtrustSrvr {
 private:
  RdmaOps &rdma_ops;
  IRegDatasetMngr &mngr;
  TrainInputParams t_params;
  RegMemSrvr &regMem;
  std::vector<ClientDataRbyz> &clnt_data_vec;
  std::vector<std::vector<int>> step_times;
  std::vector<float> log_TS_vec;

  std::vector<int> generateRandomUniqueVector(int n_clients, int min_sz);

  torch::Tensor aggregateUpdates(const std::vector<torch::Tensor> &client_updates,
                                 const torch::Tensor &server_update,
                                 const std::vector<uint32_t> &clnt_indices);

  bool expBackoffWait(int round, int client);

 public:
  FLtrustSrvr(RdmaOps &rdma_ops, IRegDatasetMngr &mngr, TrainInputParams t_params,
              std::vector<ClientDataRbyz> &clnt_data_vec, RegMemSrvr &regMem)
      : rdma_ops(rdma_ops),
        mngr(mngr),
        t_params(t_params),
        clnt_data_vec(clnt_data_vec),
        regMem(regMem),
        log_TS_vec(t_params.n_clients, 0.0f) {}

  std::vector<torch::Tensor> run();
};