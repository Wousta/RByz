#include "rbyzAux.hpp"
#include "datasetLogic/registeredMnistTrain.hpp"
#include "global/globalConstants.hpp"
#include "global/logger.hpp"
#include "tensorOps.hpp"

#include <algorithm>

void readClntsRByz(int n_clients, RdmaOps &rdma_ops, std::vector<std::atomic<int>> &clnt_CAS) {
  int clnt_idx = 0;
  while (clnt_idx < n_clients) {
    rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_FREE, MEM_OCCUPIED, clnt_idx);

    if (clnt_CAS[clnt_idx].load() == MEM_OCCUPIED) {
      std::this_thread::yield();
    } else {
      // Read the data from the client and release client lock
      rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOSS_AND_ERR_IDX, clnt_idx);
      clnt_CAS[clnt_idx].store(MEM_FREE);
      rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_OCCUPIED, MEM_FREE, clnt_idx);
      clnt_idx++;
    }
  }

  Logger::instance().log("Server: All clients read\n");
}

void aquireCASLock(int clnt_idx, RdmaOps &rdma_ops, std::vector<std::atomic<int>> &clnt_CAS) {
  int current = MEM_OCCUPIED;
  while (current == MEM_OCCUPIED) {
    rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_FREE, MEM_OCCUPIED, clnt_idx);
    current = clnt_CAS[clnt_idx].load();

    if (current == MEM_OCCUPIED) {
      std::this_thread::yield();
    }
  }
}

void releaseCASLock(int clnt_idx, RdmaOps &rdma_ops, std::vector<std::atomic<int>> &clnt_CAS) {
  clnt_CAS[clnt_idx].store(MEM_FREE);
  rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_OCCUPIED, MEM_FREE, clnt_idx);
}

void updateTS(std::vector<ClientDataRbyz> &clnt_data_vec,
              ClientDataRbyz &clnt_data,
              float srvr_loss,
              float srvr_error_rate) {
  // loss and error of client to update
  float w_loss = *clnt_data.loss;
  float w_err = *clnt_data.error_rate;

  // Find minimum loss and error among all clients
  float min_w_loss = FLT_MAX;
  float min_w_err = FLT_MAX;
  for (ClientDataRbyz clnt_data : clnt_data_vec) {
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

void writeErrorAndLoss(BaseMnistTrain& mnist, float* loss_and_err) {
  float loss_val = mnist.getLoss();
  float error_rate_val = mnist.getErrorRate();
  std::memcpy(loss_and_err, &loss_val, sizeof(float));
  std::memcpy(loss_and_err + 1, &error_rate_val, sizeof(float));
}

void runRByzClient(std::vector<torch::Tensor> &w,
                   std::atomic<int> &clnt_CAS,
                   RegisteredMnistTrain &mnist,
                   float *clnt_w,
                   float* loss_and_err) {
  Logger::instance().log("\n\n=============================================\n");
  Logger::instance().log("==============  STARTING RBYZ  ==============\n");
  Logger::instance().log("=============================================\n");

  // Before rbyz, the client has to write error and loss for the first time
  writeErrorAndLoss(mnist, clnt_w);
  Logger::instance().log("Client: Initial loss and error values\n");
  clnt_CAS.store(MEM_FREE);

  for (int round = 1; round < GLOBAL_ITERS_RBYZ; round++) {
    w = mnist.runMnistTrain(round, w);

    // Store the updated weights in clnt_w
    torch::Tensor all_tensors = flatten_tensor_vector(w);
    size_t total_bytes_g = all_tensors.numel() * sizeof(float);
    if(total_bytes_g != (size_t)REG_SZ_DATA) {
      Logger::instance().log("REG_SZ_DATA and total_bytes sent do not match!!\n");
    }

    float* all_tensors_float = all_tensors.data_ptr<float>();

    // Make server wait until memory is written
    int expected = MEM_FREE;
    while(!clnt_CAS.compare_exchange_strong(expected, MEM_OCCUPIED)) {
      std::this_thread::yield();
    }
    Logger::instance().log("CAS LOCK AQUIRED\n");

    // Store the updates, error and loss values in clnt_w
    std::memcpy(clnt_w, all_tensors_float, total_bytes_g);
    writeErrorAndLoss(mnist, loss_and_err);

    // Reset the memory ready flag
    clnt_CAS.store(MEM_FREE);
    Logger::instance().log("CAS LOCK RELEASED\n");
  }
}