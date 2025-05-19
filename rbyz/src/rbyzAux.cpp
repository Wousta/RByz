#include "rbyzAux.hpp"
#include "datasetLogic/registeredMnistTrain.hpp"
#include "global/globalConstants.hpp"
#include "global/logger.hpp"
#include "tensorOps.hpp"

#include <algorithm>

//////////////////////////////////////////////////////////////
////////////////////// SERVER FUNCTIONS //////////////////////
void readClntsRByz(int n_clients, RdmaOps &rdma_ops, std::vector<ClientDataRbyz> &clnt_data_vec) {
  int clnt_idx = 0;
  while (clnt_idx < n_clients) {
    rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_FREE, MEM_OCCUPIED, clnt_idx);
    
    std::atomic<int> &clnt_CAS = clnt_data_vec[clnt_idx].clnt_CAS;
    if (clnt_CAS.load() == MEM_OCCUPIED) {
      std::this_thread::yield();
    } else {
      // Read the data from the client and release client lock
      rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOSS_AND_ERR_IDX, clnt_idx);
      clnt_CAS.store(MEM_FREE);
      rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_OCCUPIED, MEM_FREE, clnt_idx);
      clnt_idx++;
    }
  }

  Logger::instance().log("Server: All clients read\n");
}

void aquireClntCASLock(int clnt_idx, RdmaOps &rdma_ops, std::atomic<int> &clnt_CAS) {
  int current = MEM_OCCUPIED;
  while (current == MEM_OCCUPIED) {
    rdma_ops.exec_rdma_CAS(sizeof(int), CLNT_CAS_IDX, MEM_FREE, MEM_OCCUPIED, clnt_idx);
    current = clnt_CAS.load();

    if (current == MEM_OCCUPIED) {
      Logger::instance().log("--------> WARNING: CAS LOCK NOT AQUIRED\n");
      std::this_thread::yield();
    }
  }
}

void releaseClntCASLock(int clnt_idx, RdmaOps &rdma_ops, std::atomic<int> &clnt_CAS) {
  clnt_CAS.store(MEM_FREE);
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
  for (ClientDataRbyz& clnt_data : clnt_data_vec) {
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

//////////////////////////////////////////////////////////////
////////////////////// CLIENT FUNCTIONS //////////////////////
void writeErrorAndLoss(BaseMnistTrain& mnist, float* loss_and_err) {
  float loss_val = mnist.getLoss();
  float error_rate_val = mnist.getErrorRate();
  std::memcpy(loss_and_err, &loss_val, sizeof(float));
  std::memcpy(loss_and_err + 1, &error_rate_val, sizeof(float));
}

void aquireCASLock(RegMemClnt& regMem) {
  int expected = MEM_FREE;
  while (!regMem.clnt_CAS.compare_exchange_strong(expected, MEM_OCCUPIED)) {
    Logger::instance().log("--------> WARNING: CAS LOCK NOT AQUIRED\n");
    std::this_thread::yield();
  }
  Logger::instance().log("CAS LOCK AQUIRED\n");
}

void releaseCASLock(RegMemClnt& regMem) {
  regMem.clnt_CAS.store(MEM_FREE);
  Logger::instance().log("CAS LOCK RELEASED\n");
}

//////////////////////////////////////////////////////////////
/////////////////////// RBYZ ALGORITHM ///////////////////////
/**
 * @brief Run the RByz client, only the clients call this function.
 */
void runRByzClient(std::vector<torch::Tensor> &w,
                   RegisteredMnistTrain &mnist,
                   RegMemClnt &regMem,
                   RdmaOps& rdma_ops) {
  Logger::instance().log("\n\n==============  STARTING RBYZ  ==============\n");

  // Wait for the server to be ready
  while (regMem.srvr_ready_flag != SRVR_READY_RBYZ) {
    std::this_thread::yield();
  }
  Logger::instance().log("Client: Server is ready\n");

  // Read the weights from the server
  rdma_ops.read_mnist_update(w, regMem.srvr_w, SRVR_W_IDX);

  // Before rbyz, the client has to write error and loss for the first time
  writeErrorAndLoss(mnist, regMem.loss_and_err);
  Logger::instance().log("Client: Initial loss and error values\n");
  regMem.clnt_CAS.store(MEM_FREE);

  for (int round = 1; round < GLOBAL_ITERS_RBYZ; round++) {

    // TODO: properly do local step logic, sampling all data before training is too slow?
    for (int step = 0; step < LOCAL_STEPS_RBYZ; step++) {
      regMem.local_step.store(round);
      w = mnist.runMnistTrain(round, w);
    }

    // Store the updated weights in clnt_w
    torch::Tensor all_tensors = flatten_tensor_vector(w);
    size_t total_bytes_g = all_tensors.numel() * sizeof(float);
    if(total_bytes_g != (size_t)REG_SZ_DATA) {
      Logger::instance().log("REG_SZ_DATA and total_bytes sent do not match!!\n");
    }

    float* all_tensors_float = all_tensors.data_ptr<float>();

    // Make server wait until memory is written
    aquireCASLock(regMem);

    // Store the updates, error and loss values in clnt_w
    std::memcpy(regMem.clnt_w, all_tensors_float, total_bytes_g);
    unsigned int total_bytes_w_int = static_cast<unsigned int>(REG_SZ_DATA);
    rdma_ops.exec_rdma_write(total_bytes_w_int, CLNT_W_IDX);
    writeErrorAndLoss(mnist, regMem.loss_and_err);
    rdma_ops.exec_rdma_write(MIN_SZ, CLNT_LOSS_AND_ERR_IDX);

    // Reset the memory ready flag
    releaseCASLock(regMem);
    
  }
}

void runRByzServer(int n_clients,
                    std::vector<torch::Tensor>& w,
                    RegisteredMnistTrain& mnist,
                    RdmaOps& rdma_ops,
                    RegMemSrvr& regMem,
                    std::vector<ClientDataRbyz>& clnt_data_vec) {
  
  Logger::instance().log("\n\n==============  STARTING RBYZ  ==============\n");
  // First write updates to server memory
  auto all_tensors = flatten_tensor_vector(w);
  size_t total_bytes = all_tensors.numel() * sizeof(float);
  float *global_w = all_tensors.data_ptr<float>();
  std::memcpy(regMem.srvr_w, global_w, total_bytes);
  
  // Signal to clients that the server is ready
  regMem.srvr_ready_flag = SRVR_READY_RBYZ;
  for (int i = 0; i < n_clients; i++) {
    rdma_ops.exec_rdma_write(MIN_SZ, SRVR_READY_IDX, i);
  }
  
  for (int round = 1; round < GLOBAL_ITERS_RBYZ; round++) {
    // Read the error and loss from the clients
    readClntsRByz(n_clients, rdma_ops, clnt_data_vec);

    // For each client run N rounds of RByz
    for (int step = 0; step < LOCAL_STEPS_RBYZ; step++) {
      int j = 0;
      while (j < n_clients) {
        // Read error and loss from client
        aquireClntCASLock(j, rdma_ops, clnt_data_vec[j].clnt_CAS);
        rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOSS_AND_ERR_IDX, j);
        releaseClntCASLock(j, rdma_ops, clnt_data_vec[j].clnt_CAS);

        if (step != 1) {
          // Run inference on server model to update its VD loss and error and then update TS
          mnist.runInference();
          updateTS(clnt_data_vec, clnt_data_vec[j], mnist.getLoss(), mnist.getErrorRate());
        }

        // Byz detection

        if (step == LOCAL_STEPS_RBYZ) {
          // Aggregate
        }

        // TODO: to avoid the server running all rounds, it has to check if the client has advanced
        j++;
      }
    }
  }
  
  // Test the model after training
  mnist.testModel();

}