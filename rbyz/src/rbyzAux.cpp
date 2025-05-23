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

bool compareVDOut(RegisteredMnistTrain& mnist, ClientDataRbyz& clnt_data) {
  // Compare the output of the forward pass with the server's output
  float* clnt_out = clnt_data.forward_pass;
  uint32_t* clnt_indices = clnt_data.forward_pass_indices;
  float* srvr_out = mnist.getForwardPass();
  uint32_t* srvr_indices = mnist.getForwardPassIndices();
  size_t clnt_forward_pass_size = clnt_data.forward_pass_mem_size / sizeof(float);
  
  size_t clnt_num_indices = clnt_data.forward_pass_indices_mem_size / sizeof(uint32_t);
  size_t loss_idx = 0;
  size_t error_idx = clnt_forward_pass_size / 2;

  if (clnt_forward_pass_size != clnt_num_indices) {
    throw std::runtime_error("compareVDOut: Forward pass and indices sizes do not match");
  }

  std::unordered_set<uint32_t> inserted_indices_set;
  for(size_t srvr_idx : clnt_data.inserted_indices) {
    inserted_indices_set.insert(*mnist.getOriginalIndex(srvr_idx));
  }

  size_t srvr_num_indices = mnist.getForwardPassMemSize() / sizeof(float);
  std::unordered_map<uint32_t, int> srvr_indices_map;
  for (size_t i = 0; i < srvr_num_indices; ++i) {
    srvr_indices_map[srvr_indices[i]] = i;
  }

  // Tolerance threshold for comparing floating point values
  const float TOLERANCE = 1e-4;
  bool validation_passed = true;
  
    // Check only the inserted indices (VD samples) to minimize comparisons
  for (size_t i = 0; i < clnt_num_indices && validation_passed; ++i) {
    uint32_t clnt_idx = clnt_indices[i];
    
    // Only compare samples that were inserted by the server
    if (inserted_indices_set.find(clnt_idx) != inserted_indices_set.end()) {
      // Find corresponding server index
      auto srvr_it = srvr_indices_map.find(clnt_idx);
      if (srvr_it == srvr_indices_map.end()) {
        // This should never happen for inserted indices
        Logger::instance().log("Missing server index for inserted sample: " + std::to_string(clnt_idx) + "\n");
        return false;
      }
      
      size_t srvr_i = srvr_it->second;
      
      // Compare loss values
      float clnt_loss = clnt_out[loss_idx + i];
      float srvr_loss = srvr_out[loss_idx + srvr_i];
      
      // Compare error values
      float clnt_error = clnt_out[error_idx + i];
      float srvr_error = srvr_out[error_idx + srvr_i];
      
      // Check if values match within tolerance
      bool loss_match = std::abs(clnt_loss - srvr_loss) < TOLERANCE;
      bool error_match = std::abs(clnt_error - srvr_error) < TOLERANCE;
      
      if (!loss_match || !error_match) {
        // Log the mismatch and fail immediately
        Logger::instance().log("Mismatch at index " + std::to_string(clnt_idx) + 
                              " - Loss: client=" + std::to_string(clnt_loss) + 
                              ", server=" + std::to_string(srvr_loss) +
                              " | Error: client=" + std::to_string(clnt_error) + 
                              ", server=" + std::to_string(srvr_error) + "\n");
        validation_passed = false;
      }
    }
  }
  
  // All comparisons passed
  Logger::instance().log("VD validation passed for all " + std::to_string(inserted_indices_set.size()) + " inserted samples\n");
  return validation_passed;

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
  
  // Before rbyz, the client has to write error and loss for the first time
  writeErrorAndLoss(mnist, regMem.loss_and_err);
  regMem.clnt_CAS.store(MEM_FREE);

  // Wait for the server to be ready
  while (regMem.srvr_ready_flag != SRVR_READY_RBYZ) {
    std::this_thread::yield();
  }
  Logger::instance().log("Client: Server is ready\n");

  // Read the weights from the server
  rdma_ops.read_mnist_update(w, regMem.srvr_w, SRVR_W_IDX);

  while (regMem.round.load() < GLOBAL_ITERS_RBYZ) {
    regMem.local_step.store(0);

    // TODO: properly do local step logic, sampling all data before training is too slow?
    while (regMem.local_step.load() < LOCAL_STEPS_RBYZ) {
      int step = regMem.local_step.load();
      w = mnist.runMnistTrain(step, w);
      writeErrorAndLoss(mnist, regMem.loss_and_err);

      regMem.local_step.store(step + 1);
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

    // Reset the memory ready flag
    releaseCASLock(regMem);

    regMem.round.store(regMem.round.load() + 1);
  }

  // Notify the server that the client is done
  regMem.round.store(GLOBAL_ITERS_RBYZ);
  rdma_ops.exec_rdma_write(MIN_SZ, CLNT_ROUND_IDX);
  Logger::instance().log("Client: Finished RByz\n");
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

  // Create VD splits and do first write of VD to the clients
  RegMnistSplitter reg_mnist_splitter(n_clients, mnist, clnt_data_vec);
  Logger::instance().log("Server: VD split created\n");
  std::vector<int> derangement = reg_mnist_splitter.generateDerangement();
  Logger::instance().log("Server: Derangement generated\n");

  // Send the VD samples to the clients
  for (int i = 0; i < n_clients; i++) {
    std::vector<size_t> srvr_indices = reg_mnist_splitter.getServerIndices(i, derangement);
    std::vector<size_t> clnt_offsets = reg_mnist_splitter.getClientOffsets(clnt_data_vec[i], derangement);

    // Save the chosen indices for the client for future inspection
    clnt_data_vec[i].inserted_indices = srvr_indices;

    Logger::instance().log("srvr_indices size: " + std::to_string(srvr_indices.size()) + "\n");
    Logger::instance().log("clnt_offsets size: " + std::to_string(clnt_offsets.size()) + "\n");

    // Make sure we only use the minimum number of samples that both vectors can handle
    size_t num_samples_to_send = std::min(srvr_indices.size(), clnt_offsets.size());
    Logger::instance().log("Server: num_samples_to_send = " + std::to_string(num_samples_to_send) + "\n");

    if (srvr_indices.size() != clnt_offsets.size()) {
      Logger::instance().log("Server: WARNING: server and client indices sizes do not match\n");
    }

    Logger::instance().log("Server: Sending VD samples to client " + std::to_string(i) + "\n");
    for (size_t j = 0; j < num_samples_to_send; j++) {
      size_t srvr_idx = srvr_indices[j];
      size_t clnt_offset = clnt_offsets[j];


      // Copy server sample to registered memory
      if (j % 100 == 0) {
        Logger::instance().log("Server: Copying sample " + std::to_string(srvr_idx) + "\n");
      }
      void* srvr_sample = mnist.getSample(srvr_idx);
      std::memcpy(regMem.vd_sample, srvr_sample, mnist.getSampleSize());
      
      LocalInfo local_info;
      local_info.indices.push_back(SRVR_VD_SAMPLE_IDX);
      RemoteInfo remote_info;
      remote_info.indx = CLNT_DATASET_IDX;
      remote_info.off = clnt_offset;

      if (j % 100 == 0) {
        Logger::instance().log("Server: Sending sample " + std::to_string(srvr_idx) + " to client " + std::to_string(i) + "\n");
      }
      rdma_ops.exec_rdma_write(mnist.getSampleSize(), local_info, remote_info, i, false);
    }
  }
  
  // Signal to clients that the server is ready
  regMem.srvr_ready_flag = SRVR_READY_RBYZ;
  for (int i = 0; i < n_clients; i++) {
    rdma_ops.exec_rdma_write(MIN_SZ, SRVR_READY_IDX, i);
  }
  Logger::instance().log("Server: All clients notified\n");
  
  // Control of exponential backoff
  std::chrono::milliseconds default_step_time(30000);

  for (int round = 1; round < GLOBAL_ITERS_RBYZ; round++) {
    // Read the error and loss from the clients
    readClntsRByz(n_clients, rdma_ops, clnt_data_vec);

    // For each client run N rounds of RByz
    for (int step = 0; step < LOCAL_STEPS_RBYZ; step++) {
      for (ClientDataRbyz& clnt_data : clnt_data_vec) {
        int j = clnt_data.clnt_index;      

        if (clnt_data.is_byzantine) {
          continue;
        }

        // Get current client's local step
        rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOCAL_STEP_IDX, j);
        int clnt_curr_step = clnt_data.local_step;
        Logger::instance().log("Server: Client " + std::to_string(j) + " local step = " + std::to_string(clnt_curr_step) + "\n");

        // Wait for client to finish first step with exponential backoff
        std::chrono::milliseconds initial_time(100);
        while (clnt_data.local_step != clnt_curr_step + 1) {
          std::this_thread::sleep_for(initial_time);
          initial_time *= 2;
          if (initial_time > default_step_time * 2) {
            clnt_data.is_byzantine = true;
            Logger::instance().log("Server: Client " + std::to_string(j) + " is Byzantine\n");
            break;
          }

          rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOCAL_STEP_IDX, j);
        }

        clnt_curr_step = clnt_data.local_step;

        // Test VD
        if (clnt_curr_step == 1) {
          rdma_ops.exec_rdma_read(clnt_data.forward_pass_mem_size, CLNT_FORWARD_PASS_IDX, j);
          rdma_ops.exec_rdma_read(clnt_data.forward_pass_indices_mem_size, CLNT_FORWARD_PASS_INDICES_IDX, j);
          mnist.runInference();
        }

        // Read error and loss from client
        aquireClntCASLock(j, rdma_ops, clnt_data_vec[j].clnt_CAS);
        rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOSS_AND_ERR_IDX, j);
        releaseClntCASLock(j, rdma_ops, clnt_data_vec[j].clnt_CAS);

        if (step != 1) {
          // Run inference on server model to update its VD loss and error and then update TS
          mnist.runInference();
          updateTS(clnt_data_vec, clnt_data_vec[j], mnist.getLoss(), mnist.getErrorRate());
        }

        if (step == LOCAL_STEPS_RBYZ) {
          // Aggregate
        }
      }
    }
  }

  // TODO: Wait for all clients to finish by reading their round
  Logger::instance().log("Server: Waiting for all clients to complete RByz...\n");
  for (int i = 0; i < n_clients; i++) {
    bool client_done = false;
    while (!client_done) {
      // Check if client has reached final round
      if (clnt_data_vec[i].round == GLOBAL_ITERS_RBYZ) {
        client_done = true;
        Logger::instance().log("Server: Client " + std::to_string(i) + " has completed all iterations\n");
      } else {
        std::this_thread::yield();
      }
    }
  }
  
  // Test the model after training
  mnist.testModel();

}