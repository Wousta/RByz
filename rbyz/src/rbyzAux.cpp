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
    if (w_loss < min_w_loss) {
      min_w_loss = w_loss;
    }

    if (w_err < min_w_err) {
      min_w_err = w_err;
    }
  }

  float no_bias = 0.01;  // Tweak until satisfied
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

void writeServerVD(void* vd_sample,
                   RegMnistSplitter& splitter, 
                   RegisteredMnistTrain& mnist,
                   RdmaOps& rdma_ops,
                   std::vector<ClientDataRbyz>& clnt_data_vec) {

  std::vector<int> derangement = splitter.generateDerangement();
  Logger::instance().log("Server: Derangement generated\n");

  // Send the VD samples to the clients
  for (int i = 0; i < clnt_data_vec.size(); i++) {
    if (clnt_data_vec[i].is_byzantine) {
      Logger::instance().log("Server: Client " + std::to_string(i) + " is Byzantine, skipping VD sample sending\n");
      continue;
    }

    std::vector<size_t> srvr_indices = splitter.getServerIndices(i, derangement);
    std::vector<size_t> clnt_offsets = splitter.getClientOffsets(clnt_data_vec[i], derangement);

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
      if (j % 1 == 0) {
        uint64_t srvr_label = *mnist.getLabel(srvr_idx);
        Logger::instance().log("Server: Copying sample " + std::to_string(srvr_idx) + " with label " + std::to_string(srvr_label) + "\n");
      }
      void* srvr_sample = mnist.getSample(srvr_idx);
      std::memcpy(vd_sample, srvr_sample, mnist.getSampleSize());
      
      LocalInfo local_info;
      local_info.indices.push_back(SRVR_VD_SAMPLE_IDX);
      RemoteInfo remote_info;
      remote_info.indx = CLNT_DATASET_IDX;
      remote_info.off = clnt_offset;

      Logger::instance().log("Server: Writing sample with offset " + std::to_string(clnt_offset) + "\n");

      if (j % 1 == 0) {
        Logger::instance().log("Server: Sending sample " + std::to_string(srvr_idx) + " to client " + std::to_string(i) + "\n");
      }
      rdma_ops.exec_rdma_write(mnist.getSampleSize(), local_info, remote_info, i, true);
    }
  }
  Logger::instance().log("Server: VD samples sent to all clients\n");
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

  // Divided by two because forward pass output contains both loss and error rate
  if (clnt_forward_pass_size / 2 != clnt_num_indices) {
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
  Logger::instance().log("Client: Initial Error and loss from client: " +
                         std::to_string(*regMem.loss_and_err) + ", " +
                         std::to_string(*(regMem.loss_and_err + 1)) + "\n");
                         
  regMem.clnt_CAS.store(MEM_FREE);

  Logger::instance().log("Srvr ready flag: " + std::to_string(regMem.srvr_ready_flag) + "\n");

  while (regMem.round.load() < GLOBAL_ITERS_RBYZ) {
    Logger::instance().log("\n//////////////// Client: Round " + std::to_string(regMem.round.load()) + " started ////////////////\n");
    // Wait for the server to be ready
    while (regMem.srvr_ready_flag != regMem.round.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Read the aggregated weights from the server
    rdma_ops.read_mnist_update(w, regMem.srvr_w, SRVR_W_IDX);

    // Run local training steps, all training data is sampled before training to let server insert VD samples
    regMem.local_step.store(0);
    while (regMem.local_step.load() < LOCAL_STEPS_RBYZ) {
      int step = regMem.local_step.load();
      Logger::instance().log("  Client: Running step " + std::to_string(step) + " of RByz\n");
      w = mnist.runMnistTrain(step, w);
      writeErrorAndLoss(mnist, regMem.loss_and_err);
      Logger::instance().log("    -> Error and loss written: " +
                        std::to_string(*regMem.loss_and_err) + ", " +
                        std::to_string(*(regMem.loss_and_err + 1)) + "\n");
      regMem.local_step.store(step + 1);
    }

    // Store the updated weights in clnt_w
    torch::Tensor all_tensors = flatten_tensor_vector(w);
    size_t total_bytes_g = all_tensors.numel() * sizeof(float);
    if(total_bytes_g != (size_t)REG_SZ_DATA) {
      Logger::instance().log("  REG_SZ_DATA and total_bytes sent do not match!!\n");
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

    std::cout << "\n//////////////// Client: Round " << regMem.round.load() << " completed ////////////////\\n";
    Logger::instance().log("\n//////////////// Client: Round " + std::to_string(regMem.round.load()) + " completed ////////////////\n");
    regMem.round.store(regMem.round.load() + 1);
    rdma_ops.exec_rdma_write(MIN_SZ, CLNT_ROUND_IDX);
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

  // Create VD splits and do first write of VD to the clients
  RegMnistSplitter splitter(n_clients, mnist, clnt_data_vec);
  
  // RBYZ training loop
  for (int round = 0; round < GLOBAL_ITERS_RBYZ; round++) {
    auto all_tensors = flatten_tensor_vector(w);
    size_t total_bytes = all_tensors.numel() * sizeof(float);
    float *global_w = all_tensors.data_ptr<float>();
    std::memcpy(regMem.srvr_w, global_w, total_bytes);

    // Read the error and loss from the clients
    readClntsRByz(n_clients, rdma_ops, clnt_data_vec);
    Logger::instance().log("Error and loss from client:\n");
    for (ClientDataRbyz& clnt_data : clnt_data_vec) {
      Logger::instance().log("Client " + std::to_string(clnt_data.index) + 
                             " loss = " + std::to_string(*clnt_data.loss) + 
                             ", error = " + std::to_string(*clnt_data.error_rate) + "\n");
    }

    // Before each round, write the server's VD to the clients to test after first local step
    //writeServerVD(regMem.vd_sample, splitter, mnist, rdma_ops, clnt_data_vec);
  
    // Signal to clients that the server is ready
    regMem.srvr_ready_flag = round;
    Logger::instance().log("Server: wrote ready flag: " + std::to_string(regMem.srvr_ready_flag) + "\n");
    for (int i = 0; i < n_clients; i++) {
      rdma_ops.exec_rdma_write(MIN_SZ, SRVR_READY_IDX, i);
    }
    Logger::instance().log("Server: All clients notified\n");

    // For each client run N rounds of RByz
    for (int srvr_step = 0; srvr_step < LOCAL_STEPS_RBYZ; srvr_step++) {
      Logger::instance().log("  Server: Running step " + std::to_string(srvr_step) + " of RByz\n");
      for (ClientDataRbyz& clnt_data : clnt_data_vec) {
        int j = clnt_data.index;      

        if (clnt_data.is_byzantine) {
          continue;
        }

        // Get current client's local step and round
        rdma_ops.exec_rdma_read(MIN_SZ, CLNT_ROUND_IDX, j);
        rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOCAL_STEP_IDX, j);
        Logger::instance().log("    -> Server: Client " + std::to_string(j) + " local step = " + std::to_string(clnt_data.local_step) + " server step = " + std::to_string(srvr_step) + "\n");

        // Wait for client to finish the step with exponential backoff
        std::chrono::milliseconds default_step_time(30000);
        std::chrono::milliseconds initial_time(100);

        while (clnt_data.local_step != srvr_step + 1 && clnt_data.local_step != LOCAL_STEPS_RBYZ) {
          std::this_thread::sleep_for(initial_time);
          initial_time *= 2;
          if (initial_time > default_step_time * 2) {
            clnt_data.is_byzantine = true;
            Logger::instance().log("    -> Server waiting: Client " + std::to_string(j) + " is Byzantine\n");
            break;
          }

          rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOCAL_STEP_IDX, j);
        }
        Logger::instance().log("    -> Server: Client " + std::to_string(j) + " now in step " + std::to_string(clnt_data.local_step) + "\n");

        // Test VD
        // if (clnt_data.local_step == 1) {
        //   rdma_ops.exec_rdma_read(clnt_data.forward_pass_mem_size, CLNT_FORWARD_PASS_IDX, j);
        //   rdma_ops.exec_rdma_read(clnt_data.forward_pass_indices_mem_size, CLNT_FORWARD_PASS_INDICES_IDX, j);
        //   mnist.runInference();
        //   if (!compareVDOut(mnist, clnt_data)) {
        //     Logger::instance().log("Server: VD validation failed for client " + std::to_string(j) + "\n");
        //     clnt_data.is_byzantine = true;
        //     continue;
        //   } else {
        //     Logger::instance().log("Server: VD validation passed for client " + std::to_string(j) + "\n");
        //   }
        // }

        // Read error and loss from client
        aquireClntCASLock(j, rdma_ops, clnt_data_vec[j].clnt_CAS);
        rdma_ops.exec_rdma_read(MIN_SZ, CLNT_LOSS_AND_ERR_IDX, j);
        Logger::instance().log("    -> Server: Read loss and error for client " + std::to_string(j) + ": " +
                               std::to_string(*clnt_data_vec[j].loss) + ", " +
                               std::to_string(*clnt_data_vec[j].error_rate) + "\n");
        releaseClntCASLock(j, rdma_ops, clnt_data_vec[j].clnt_CAS);

        Logger::instance().log("      -> Read loss and error\n");

        if (clnt_data.local_step != 1) {
          // Run inference on server model to update its VD loss and error and then update TS
          Logger::instance().log("    -> Server: Running inference for client " + std::to_string(j) + "\n");
          mnist.runInference();
          updateTS(clnt_data_vec, clnt_data_vec[j], mnist.getLoss(), mnist.getErrorRate());
        }
      }

      Logger::instance().log("    Server: Completed step " + std::to_string(srvr_step) + " of RByz\n");
    }

    // Read client updates and aggregate them
    std::vector<torch::Tensor> clnt_updates;
    std::vector<uint32_t> clnt_indices;
    clnt_updates.reserve(n_clients);
    
    for (size_t i = 0; i < n_clients; i++) {
      ClientDataRbyz& client = clnt_data_vec[i];
      if (client.is_byzantine) {
        continue;
      }

      Logger::instance().log("reading flags from client: " + std::to_string(i) + "\n");
      while (client.round != round + 1) {
        std::this_thread::yield();
      }

      size_t numel_server = REG_SZ_DATA / sizeof(float);
      torch::Tensor flat_tensor =
          torch::from_blob(
              regMem.clnt_ws[i], {static_cast<long>(numel_server)}, torch::kFloat32)
              .clone();

      clnt_updates.push_back(flat_tensor);
      clnt_indices.push_back(client.index);
    }

    // Use attacks to simulate Byzantine clients
    clnt_updates = no_byz(clnt_updates, mnist.getModel(), GLOBAL_LEARN_RATE, N_BYZ_CLNTS, mnist.getDevice());

    // Aggregation
    for (int i = 0; i < n_clients; i++) {
      int clnt_idx = clnt_indices[i];

      if (clnt_updates[i].numel() != REG_SZ_DATA / sizeof(float)) {
        throw std::runtime_error("Server: Client update size does not match expected size");
      }

      for (size_t j = 0; j < w.size(); j++) {
        w[j] = w[j] + clnt_updates[i][j] * clnt_data_vec[clnt_idx].trust_score * GLOBAL_LEARN_RATE;
      }

      Logger::instance().log("Trust score for client " + std::to_string(clnt_idx) + ": " + 
                             std::to_string(clnt_data_vec[clnt_idx].trust_score) + "\n");
    }

    std::cout << "\n///////////////// Server: Round " << round << " completed /////////////////\n";
    Logger::instance().log("\n//////////////// Server: Round " + std::to_string(round) + " completed ////////////////\n");
    mnist.testModel();
  }

  // TODO: Wait for all clients to finish by reading their round
  Logger::instance().log("Server: Waiting for all clients to complete RByz...\n");
  for (int i = 0; i < n_clients; i++) {
    if (clnt_data_vec[i].is_byzantine) {
      continue;
    }
    
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