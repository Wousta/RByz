#include "../include/rbyzAux.hpp"

#include <algorithm>

void readClntsRByz(
    int n_clients,
    RdmaOps& rdma_ops,
    std::vector<std::atomic<int>>& clnt_CAS) {
  
    int clnt_idx = 0;
    while (clnt_idx < n_clients) {
      rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_FREE, MEM_OCCUPIED, clnt_idx);

      if (clnt_CAS[clnt_idx].load() == MEM_OCCUPIED) {
        std::this_thread::yield();
      }
      else {
        // Read the data from the client and release client lock
        rdma_ops.exec_rdma_read(REG_SZ_CLNT, CLNT_W_IDX, clnt_idx);
        clnt_CAS[clnt_idx].store(MEM_FREE);
        rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_OCCUPIED, MEM_FREE, clnt_idx);
        clnt_idx++;
      }
    }

    Logger::instance().log("Server: All clients read\n");
}

void aquireCASLock(
  int clnt_idx, 
  RdmaOps& rdma_ops,
  std::vector<std::atomic<int>>& clnt_CAS) {

  int current = MEM_OCCUPIED;
  while (current == MEM_OCCUPIED) {
    rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_FREE, MEM_OCCUPIED, clnt_idx);
    current = clnt_CAS[clnt_idx].load();

    if (current == MEM_OCCUPIED) {
      std::this_thread::yield();
    }
  }
}

void releaseCASLock(
  int clnt_idx, 
  RdmaOps& rdma_ops,
  std::vector<std::atomic<int>>& clnt_CAS) {
    
  clnt_CAS[clnt_idx].store(MEM_FREE);
  rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_OCCUPIED, MEM_FREE, clnt_idx);
}

void updateTS(
    std::vector<ClientData>& clnt_data_vec,
    ClientData& clnt_data, 
    float srvr_loss, 
    float srvr_error_rate) {

    // loss and error of client to update
    float w_loss = *clnt_data.loss;
    float w_err = *clnt_data.error_rate;
  
    // Find minimum loss and error among all clients
    float min_w_loss = FLT_MAX;
    float min_w_err = FLT_MAX;
    for (ClientData clnt_data : clnt_data_vec) {
      if (*clnt_data.loss < min_w_loss) {
        min_w_loss = *clnt_data.loss;
      }

      if (*clnt_data.error_rate < min_w_err) {
        min_w_err = *clnt_data.error_rate;
      }
    }

    float no_bias = 1.0;  // Tweak until satisfied
    float max_loss = std::max(0.0f, srvr_loss - w_loss);
    float loss_Calc = max_loss / (srvr_loss - min_w_loss + no_bias);
    float max_err = std::max(0.0f, srvr_error_rate - w_err);
    float err_Calc = max_err / (srvr_error_rate - min_w_err + no_bias);

    clnt_data.trust_score = (loss_Calc + err_Calc) / 2;
}