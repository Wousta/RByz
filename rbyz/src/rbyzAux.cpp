#include "rbyzAux.hpp"
#include "attacks.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"

#include <algorithm>
#include <unistd.h>


//////////////////////////////////////////////////////////////
////////////////////// SERVER FUNCTIONS //////////////////////
void RByzAux::awaitTermination(std::vector<ClientDataRbyz>& clnt_data_vec, int rounds_rbyz) {
  Logger::instance().log("Server: Waiting for all clients to complete RByz...\n");
  for (int i = 0; i < clnt_data_vec.size(); i++) {
    if (clnt_data_vec[i].is_byzantine) {
      continue;
    }
    
    bool client_done = false;
    while (!client_done) {
      // Check if client has reached final round
      if (clnt_data_vec[i].round == rounds_rbyz) {
        client_done = true;
        Logger::instance().log("Server: Client " + std::to_string(i) + " has completed all iterations\n");
      } else {
        std::this_thread::yield();
      }
    }
  }
}

void RByzAux::updateTS(std::vector<ClientDataRbyz> &clnt_data_vec,
              ClientDataRbyz &clnt_data,
              float srvr_loss,
              float srvr_error_rate) {
  // loss and error of client to update
  float w_loss = clnt_data.loss_clnt;
  float w_err = clnt_data.error_rate_clnt;

  Logger::instance().log("        -> Client " + std::to_string(clnt_data.index) + 
                         " loss: " + std::to_string(w_loss) + 
                         ", error rate: " + std::to_string(w_err) + "\n");

  // Find minimum loss and error among all clients
  float min_w_loss = w_loss;
  float min_w_err = w_err;
  for (ClientDataRbyz& clnt_data : clnt_data_vec) {
    if (*clnt_data.loss < min_w_loss) {
      min_w_loss = *clnt_data.loss;
    }

    if (*clnt_data.error_rate < min_w_err) {
      min_w_err = *clnt_data.error_rate;
    }
  }

  Logger::instance().log("        -> Minimum loss: " + std::to_string(min_w_loss) + 
                         ", Minimum error rate: " + std::to_string(min_w_err) + "\n");

  float no_bias = 0.01;  // Tweak until satisfied
  float max_loss = std::max(0.0f, srvr_loss - w_loss);
  float loss_Calc = max_loss / (srvr_loss - min_w_loss + no_bias);
  float max_err = std::max(0.0f, srvr_error_rate - w_err);
  float err_Calc = max_err / (srvr_error_rate - min_w_err + no_bias);

  Logger::instance().log("        -> Server loss: " + std::to_string(srvr_loss) + 
                         ", Server error rate: " + std::to_string(srvr_error_rate) + "\n");
  Logger::instance().log("        -> Max loss: " + std::to_string(max_loss) + 
                         ", Max error rate: " + std::to_string(max_err) + "\n");
  Logger::instance().log("        -> Calculated loss_Calc: " + std::to_string(loss_Calc) + 
                         ", err_Calc: " + std::to_string(err_Calc) + "\n");

  clnt_data.trust_score = (loss_Calc + err_Calc) / 2;
}

torch::Tensor RByzAux::aggregate_updates(const std::vector<torch::Tensor>& client_updates,
                                const torch::Tensor& flat_w,
                                const std::vector<ClientDataRbyz> &clnt_data_vec,
                                const std::vector<uint32_t>& clnt_indices) {

  // Normalize the client updates
  std::vector<torch::Tensor> normalized_updates;
  normalized_updates.reserve(client_updates.size());
  float server_norm = torch::norm(flat_w, 2).item<float>();

  Logger::instance().log("\nComputing aggregation data ================\n");

  for (const auto &client_update : client_updates) {
    float client_norm = torch::norm(client_update, 2).item<float>();
    torch::Tensor normalized_update = client_update * (server_norm / client_norm);
    normalized_updates.push_back(normalized_update);
  }

  float trust_scores[client_updates.size()];
  for (int i = 0; i < client_updates.size(); i++) {
    // Byzantine clients are skipped, so we have to keep track of good clients indices clnt_indices[i]
    trust_scores[i] = clnt_data_vec[clnt_indices[i]].trust_score;
  }

  // Normalize trust scores
  float sum_trust = 0.0f;
  for (float score : trust_scores) {
    sum_trust += score;
  }

  torch::Tensor aggregated_update = torch::zeros_like(flat_w);
  if (sum_trust > 0) {
    for (int i = 0; i < client_updates.size(); i++) {
      aggregated_update += normalized_updates[i] * trust_scores[i];
    }

    aggregated_update /= sum_trust;
  }

  Logger::instance().log("Final Aggregated update: " + aggregated_update.slice(0, 0, std::min<size_t>(aggregated_update.numel(), 5)).toString() + "\n");
  Logger::instance().log("================================\n");

  return aggregated_update;
}

void RByzAux::writeServerVD(RegMnistSplitter& splitter,
                            std::vector<ClientDataRbyz>& clnt_data_vec) {

  std::vector<int> derangement = splitter.generateDerangement();

  // Send the VD samples to the clients
  for (int i = 0; i < clnt_data_vec.size(); i++) {
    if (clnt_data_vec[i].is_byzantine) {
      Logger::instance().log("Server: Client " + std::to_string(i) + " is Byzantine, skipping VD sample sending\n");
      continue;
    }

    std::vector<size_t> srvr_indices = splitter.getServerIndices(i, derangement);
    std::vector<size_t> clnt_chunks = splitter.getClientChunks(clnt_data_vec[i].index);

    size_t srvr_idx = 0;
    for (size_t j = 0; j < clnt_chunks.size(); j++) {

      size_t clnt_chunk = clnt_chunks[j];

      LocalInfo local_info;
      RemoteInfo remote_info;
      remote_info.indx = REG_DATASET_IDX;
      remote_info.off = clnt_chunk;

      for (int k = 0; k < splitter.getSamplesPerChunk(); k++) {
        if (srvr_idx >= srvr_indices.size()) {
          Logger::instance().log("Warning: Not enough VD samples for client " + std::to_string(i) + "\n");
          break;
        }
        size_t srvr_sample_idx = srvr_indices[srvr_idx++];
        clnt_data_vec[i].inserted_indices.insert(srvr_sample_idx);
        uint64_t sample_offset = mngr.getSampleOffset(srvr_sample_idx);
        local_info.indices.push_back(REG_DATASET_IDX);
        local_info.offs.push_back(sample_offset);
      }
      rdma_ops.exec_rdma_write(splitter.getChunkSize(), local_info, remote_info, i, false);
    }
  }
  Logger::instance().log("Server: VD samples sent to all clients\n");
}

bool RByzAux::processVDOut(ClientDataRbyz& clnt_data, bool check_byz) {
  // Compare the output of the forward pass with the server's output
  float* clnt_out = clnt_data.forward_pass;
  uint32_t* clnt_indices = clnt_data.forward_pass_indices;
  float* srvr_out = mngr.f_pass_data.forward_pass;
  uint32_t* srvr_indices = mngr.f_pass_data.forward_pass_indices;
  size_t clnt_forward_pass_size = clnt_data.forward_pass_mem_size / sizeof(float);
  
  size_t clnt_num_indices = clnt_data.forward_pass_indices_mem_size / sizeof(uint32_t);
  size_t srvr_num_indices = mngr.f_pass_data.forward_pass_indices_mem_size / sizeof(uint32_t);
  size_t clnt_error_start = clnt_forward_pass_size / 2;
  
  // Calculate server error start position
  size_t srvr_forward_pass_size = mngr.f_pass_data.forward_pass_mem_size / sizeof(float);
  size_t srvr_error_start = srvr_forward_pass_size / 2;

  // Divided by two because forward pass output contains both loss and error rate
  if (clnt_forward_pass_size / 2 != clnt_num_indices) {
    throw std::runtime_error("processVDOut: Forward pass and indices sizes do not match");
  } 

  // get the original indices of the inserted samples
  std::unordered_set<uint32_t> inserted_indices_set;
  for(size_t srvr_idx : clnt_data.inserted_indices) {
    inserted_indices_set.insert(*mngr.getOriginalIndex(srvr_idx));
  }

  if (inserted_indices_set.size() != clnt_data.inserted_indices.size()) {
    throw std::runtime_error("processVDOut: Inserted indices set size does not match the number of inserted indices");
  }

  Logger::instance().log("Server: Comparing VD output for " + std::to_string(inserted_indices_set.size()) + " inserted samples\n");

  // Map original indices of forward pass to their positions in the server output
  std::unordered_map<uint32_t, int> srvr_indices_map;
  for (size_t i = 0; i < srvr_num_indices; ++i) {
    srvr_indices_map[srvr_indices[i]] = i;
  }
  // Tolerance threshold for comparing floating point values
  const float TOLERANCE = 1e-4;
  bool validation_passed = true;
  
  float clnt_loss_total = 0.0f;
  float clnt_error_total = 0.0f;
  float srvr_loss_total = 0.0f;
  float srvr_error_total = 0.0f;
  int processed_samples = 0;
  // Check only the inserted indices (VD samples) to minimize comparisons
  for (size_t i = 0; i < clnt_num_indices && validation_passed; ++i) {
    uint32_t clnt_idx = clnt_indices[i];
    
    // Only compare samples that were inserted by the server
    if (inserted_indices_set.find(clnt_idx) != inserted_indices_set.end()) {
      // Find corresponding server index
      auto srvr_it = srvr_indices_map.find(clnt_idx);
      if (srvr_it == srvr_indices_map.end()) {
        throw std::runtime_error("Missing server index for inserted sample: " + std::to_string(clnt_idx));
      }
      
      size_t srvr_i = srvr_it->second;

      clnt_loss_total += clnt_out[i];
      srvr_loss_total += srvr_out[srvr_i];
      clnt_error_total += clnt_out[clnt_error_start + i];
      srvr_error_total += srvr_out[srvr_error_start + srvr_i];

      if (processed_samples == 0 && clnt_data.index == 0) {
        Logger::instance().log("Processing first sample of client 0: idx " + std::to_string(clnt_idx) + 
                              " loss: " + std::to_string(clnt_loss_total) + 
                              " error: " + std::to_string(clnt_error_total) + "\n");
      }

      processed_samples++;
      
      if (check_byz) {
        // Check if values match within tolerance
        float clnt_loss = clnt_out[i];
        float srvr_loss = srvr_out[srvr_i];
        float clnt_error = clnt_out[clnt_error_start + i];
        float srvr_error = srvr_out[srvr_error_start + srvr_i];

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
        } else {
          Logger::instance().log("!Match at index " + std::to_string(clnt_idx) + 
                                " - Loss: client=" + std::to_string(clnt_loss) +
                                ", server=" + std::to_string(srvr_loss) +
                                " | Error: client=" + std::to_string(clnt_error) +
                                ", server=" + std::to_string(srvr_error) + "\n");
        }
      }
    }
  }

  clnt_data.loss_clnt = clnt_loss_total / inserted_indices_set.size();
  clnt_data.error_rate_clnt = clnt_error_total / inserted_indices_set.size();
  clnt_data.loss_srvr = srvr_loss_total / inserted_indices_set.size();
  clnt_data.error_rate_srvr = srvr_error_total / inserted_indices_set.size();

  float processed_samples_proportion = static_cast<float>(processed_samples) / inserted_indices_set.size();
  if (processed_samples_proportion < 0.5) {
    Logger::instance().log("WARNING processVDOut: Client " + std::to_string(clnt_data.index) + 
                             " Processed samples below 50%: " +
                             std::to_string(processed_samples) + " vs " + 
                             std::to_string(inserted_indices_set.size()) + "\n");

    validation_passed = false;
  }

  Logger::instance().log(" Client " + std::to_string(clnt_data.index) + 
                         " VD proc results - Loss: " + std::to_string(clnt_data.loss_clnt) + 
                         ", Error Rate: " + std::to_string(clnt_data.error_rate_clnt) + 
                          " | Server Loss: " + std::to_string(clnt_data.loss_srvr) +
                          ", Server Error Rate: " + std::to_string(clnt_data.error_rate_srvr) + "\n");
  
  // All comparisons passed
  Logger::instance().log("VD validation passed for all " + std::to_string(inserted_indices_set.size()) + " inserted samples\n");
  return validation_passed;

}

void RByzAux::initTimeoutTime(std::vector<ClientDataRbyz>& clnt_data_vec) {
  for (ClientDataRbyz& clnt_data : clnt_data_vec) {
    clnt_data.limit_step_time = std::chrono::milliseconds(step_times[clnt_data.index][0]);
    Logger::instance().log("Client " + std::to_string(clnt_data.index) + 
                           " initial limit step time: " + 
                           std::to_string(clnt_data.limit_step_time.count()) + " ms\n");
  }
}

void RByzAux::runBenchMark(std::vector<ClientDataRbyz>& clnt_data_vec) {

}

//////////////////////////////////////////////////////////////
/////////////////////// RBYZ ALGORITHM ///////////////////////
void RByzAux::runRByzServer(int n_clients,
                    std::vector<torch::Tensor>& w,
                    RegMemSrvr& regMem,
                    std::vector<ClientDataRbyz>& clnt_data_vec) {
  
  Logger::instance().log("\n\n==============  STARTING RBYZ  ==============\n");
  int total_steps = t_params.global_iters_fl;

  // Initialization for timeouts
  std::vector<int> test_step(n_clients, local_steps);
  int min_steps = std::ceil(local_steps * 0.5); // Minimum steps to consider a client valid
  int middle_steps = std::ceil(local_steps * 0.75);
  std::uniform_int_distribution<int> step_range(middle_steps, local_steps);
  std::mt19937 rng(42); 

  // Create VD splits and do first write of VD to the clients
  RegMnistSplitter splitter(1, mngr, clnt_data_vec);
  
  // RBYZ training loop
  for (int round = 0; round < global_rounds; round++) {
    Logger::instance().log("\n\n=================  ROUND " + std::to_string(round) + " STARTED  =================\n");

    mngr.runTesting();

    auto flat_w = flatten_tensor_vector(w);
    size_t total_bytes = flat_w.numel() * sizeof(float);
    float *global_w = flat_w.data_ptr<float>();
    std::memcpy(regMem.srvr_w, global_w, total_bytes);

    // Before each round, write the server's VD to the clients to test after first local step
    if (round == 0) {
      writeServerVD(splitter, clnt_data_vec);
      initTimeoutTime(clnt_data_vec);
    }
  
    // Signal to clients that the server is ready
    regMem.srvr_ready_flag = round;
    Logger::instance().log("Server: wrote ready flag: " + std::to_string(regMem.srvr_ready_flag) + "\n");

    // For each client run N rounds of RByz
    for (int srvr_step = 0; srvr_step < local_steps; srvr_step++) {
      Logger::instance().log("  Server: Running step " + std::to_string(srvr_step) + " of RByz\n");

      // Log accuracy and round to Results
      Logger::instance().logRByzAcc(std::to_string(total_steps) + " " + std::to_string(mngr.test_accuracy) + "\n");
      total_steps++;

      for (ClientDataRbyz& clnt_data : clnt_data_vec) {
        int j = clnt_data.index;    

        if (clnt_data.is_byzantine || clnt_data.local_step == clnt_data.steps_to_finish) {
          continue;
        }

        rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, j);
        rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, j);
        Logger::instance().log("    -> Server: Client " + std::to_string(j) + 
                               " local step = " + std::to_string(clnt_data.local_step) + 
                               " server step = " + std::to_string(srvr_step) + 
                               " dataset size = " + std::to_string(clnt_data.dataset_size) + "\n");

        // Wait for client to finish the step with exponential backoff
        std::chrono::milliseconds initial_time(static_cast<long>(clnt_data.limit_step_time.count()));
        if (clnt_data.local_step < clnt_data.next_step && clnt_data.round == round) {
          std::this_thread::sleep_for(initial_time);
          Logger::instance().log("    -> Server waited initial: " + std::to_string(initial_time.count()) +
                                 " ms for client " + std::to_string(j) + "\n");
        }
        rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, j);
        rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, j);

        std::chrono::milliseconds exp_backoff_time(1);
        bool advanced = true;
        while (clnt_data.local_step < clnt_data.next_step && clnt_data.round == round) {
          std::this_thread::sleep_for(exp_backoff_time);
          exp_backoff_time *= 2; // Exponential backoff
          if (exp_backoff_time > clnt_data.limit_step_time) {

            // Choose lower steps to wait for and increase the limit time
            if(clnt_data.steps_to_finish <= min_steps) {
              Logger::instance().log("    -> Server: Client " + std::to_string(j) + 
                                     " is Byzantine, skipping\n");
              clnt_data.is_byzantine = true;

            } else {
              long int new_limit = static_cast<long int>(std::ceil(clnt_data.limit_step_time.count() * 1.25));
              clnt_data.limit_step_time = std::chrono::milliseconds(new_limit);
              clnt_data.steps_to_finish = step_range(rng);
              middle_steps = std::max(static_cast<int>(std::floor(clnt_data.steps_to_finish * 0.75)), min_steps);
              step_range = std::uniform_int_distribution<int>(middle_steps, clnt_data.steps_to_finish);
              Logger::instance().log("    -> Client " + std::to_string(j) + 
                                     " lowering steps to finish to: " + std::to_string(clnt_data.steps_to_finish) + "\n");

              // Client will now do steps_to_finish local steps
              clnt_data.clnt_CAS.store(clnt_data.steps_to_finish);
              rdma_ops.exec_rdma_write(sizeof(int), CLNT_CAS_IDX, j);
            }

            Logger::instance().log("    -> Server waiting: Client " + std::to_string(j) + " timed out\n");
            advanced = false;
            break;
          }
          rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, j);
        }
        Logger::instance().log("    -> Server waited: " + std::to_string(initial_time.count()) + " ns\n");

        if (advanced) {
          clnt_data.next_step = clnt_data.local_step + 1;
        }
      }
      std::cout << "\n    ---  Step " << srvr_step << " of round " << round << " completed  ---\n";
    }

    for (ClientDataRbyz& clnt_data : clnt_data_vec) {
      if (clnt_data.is_byzantine) {
        continue;
      }
      clnt_data.next_step = 1;
      rdma_ops.exec_rdma_read(clnt_data.forward_pass_mem_size, CLNT_FORWARD_PASS_IDX, clnt_data.index);
      rdma_ops.exec_rdma_read(clnt_data.forward_pass_indices_mem_size, CLNT_FORWARD_PASS_INDICES_IDX, clnt_data.index);

      // Only update TS if at least 50% of the server's VD samples were processed
      if (processVDOut(clnt_data, false)) {
        updateTS(clnt_data_vec, clnt_data, clnt_data.loss_srvr, clnt_data.error_rate_srvr);
      }
    }

    Logger::instance().log("    Trust Scores after last step of round " + std::to_string(round) + ":\n");
    for (const ClientDataRbyz& clnt_data : clnt_data_vec) {
      if (clnt_data.is_byzantine) {
        Logger::instance().log("    Client " + std::to_string(clnt_data.index) + " is Byzantine, skipping TS\n");
        continue;
      }
      Logger::instance().log("    Client " + std::to_string(clnt_data.index) +
                              " Trust Score: " + std::to_string(clnt_data.trust_score) + "\n");
    }

    std::vector<torch::Tensor> clnt_updates;
    std::vector<uint32_t> clnt_indices;
    clnt_updates.reserve(n_clients);

    // Read client updates and aggregate them
    for (size_t i = 0; i < clnt_data_vec.size(); i++) {
      ClientDataRbyz& client = clnt_data_vec[i];

      if (client.is_byzantine) {
        continue;
      }

      bool timed_out = false;
      std::chrono::microseconds initial_time(20); // time of 10 round trips
      std::chrono::microseconds limit_step_time(20000000);
      // Wait for the client to finish the round
      while (client.round != round + 1 && !timed_out) {
        std::this_thread::sleep_for(initial_time);
        initial_time *= 2; // Exponential backoff
        if (initial_time > limit_step_time) {
          timed_out = true;
          Logger::instance().log("    -> Timeout in update gathering by client " + std::to_string(client.index) + 
                                 " for round " + std::to_string(round) + "\n");
        }
      }
      Logger::instance().log("    -> Server waited: " + std::to_string(initial_time.count()) + " us for client " + 
                             std::to_string(client.index) + "\n");

      if (!timed_out) {
        size_t numel_server = regMem.reg_sz_data / sizeof(float);
        torch::Tensor flat_tensor =
            torch::from_blob(
                regMem.clnt_ws[i], {static_cast<long>(numel_server)}, torch::kFloat32)
                .clone();

        clnt_updates.push_back(flat_tensor);
        clnt_indices.push_back(client.index);
      } 
    }

    // Use attacks to simulate Byzantine clients

    // Aggregation
    torch::Tensor aggregated_update = aggregate_updates(clnt_updates, flat_w, clnt_data_vec, clnt_indices);
    std::vector<torch::Tensor> aggregated_update_vec =
        reconstruct_tensor_vector(aggregated_update, w);

    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] + t_params.global_learn_rate * aggregated_update_vec[i];
    }
    mngr.updateModelParameters(w);

    std::cout << "\n///////////////// Server: Round " << round << " completed /////////////////\n";
    Logger::instance().log("\n//////////////// Server: Round " + std::to_string(round) + " completed ////////////////\n");
  }

  awaitTermination(clnt_data_vec, global_rounds);  
}

/**
 * @brief Run the RByz client, only the clients call this function.
 */
void RByzAux::runRByzClient(std::vector<torch::Tensor> &w, RegMemClnt &regMem) {
  Logger::instance().log("\n\n==============  STARTING RBYZ  ==============\n");
  std::string log_file = "lstep_" + std::to_string(getpid()) + ".log";
  Logger::instance().logCustom("./stepTimes", log_file, std::to_string(regMem.id - 1) + "\n");
  
  while (regMem.round.load() < global_rounds) {

    if (regMem.round.load() == 2) {
      Logger::instance().log("POST: first mnist samples\n");
      for (int i = 0; i < 5; i++) {
        Logger::instance().log("Sample " + std::to_string(i) + ": label = " + std::to_string(*mngr.getLabel(i)) + 
                              " | og_idx = " + std::to_string(*mngr.getOriginalIndex(i)) + "\n");
      }
    }

    regMem.local_step.store(0);
    
    Logger::instance().log("\n//////////////// Client: Round " + std::to_string(regMem.round.load()) + " started ////////////////\n");
    // Wait for the server to be ready
    do {
      rdma_ops.exec_rdma_read(sizeof(int), SRVR_READY_IDX);
      std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    } while (regMem.srvr_ready_flag != regMem.round.load());

    // Read the aggregated weights from the server and update the local weights
    Logger::instance().log("Client: Reading server weights for round " + std::to_string(regMem.round.load()) + "\n");
    rdma_ops.read_mnist_update(w, regMem.srvr_w, regMem.reg_sz_data, SRVR_W_IDX);
    // mnist.updateModelParameters(w);
    // mnist.testModel();

    Logger::instance().log("Client: Read server weights for round " + std::to_string(regMem.round.load()) + "\n");
    // Run local training steps, all training data is sampled before training to let server insert VD samples
    Logger::instance().log("Steps to run: " + std::to_string(regMem.CAS.load()) + "\n");
    while (regMem.local_step.load() < regMem.CAS.load()) {
      // auto start = std::chrono::high_resolution_clock::now();
      int step = regMem.local_step.load();
      Logger::instance().log(" ...... Client: Running step " + std::to_string(step) + " of RByz in round " + std::to_string(regMem.round.load()) + "\n");

      w = mngr.runTraining(step, w);
      regMem.local_step.store(step + 1);
      // auto end = std::chrono::high_resolution_clock::now();
      // std::string time = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
      // Logger::instance().logCustom("./stepTimes", log_file, time + "\n");
      Logger::instance().log("Client: Local step " + std::to_string(regMem.local_step.load()) + " going ahead\n");
    }

    // Write the weights to the registered memory and write to the server
    torch::Tensor flat_w = flatten_tensor_vector(w);
    size_t total_bytes_g = flat_w.numel() * sizeof(float);
    float* flat_w_float = flat_w.data_ptr<float>();
    std::memcpy(regMem.clnt_w, flat_w_float, total_bytes_g);
    unsigned int total_bytes_w_int = static_cast<unsigned int>(regMem.reg_sz_data);
    rdma_ops.exec_rdma_write(total_bytes_w_int, CLNT_W_IDX);

    regMem.round.store(regMem.round.load() + 1);
    rdma_ops.exec_rdma_write(MIN_SZ, CLNT_ROUND_IDX);
    mngr.runTesting();
    Logger::instance().log("\n//////////////// Client: Round " + std::to_string(regMem.round.load() - 1) + " completed ////////////////\n");
  }

  // Notify the server that the client is done
  regMem.round.store(global_rounds);
  rdma_ops.exec_rdma_write(MIN_SZ, CLNT_ROUND_IDX);
  Logger::instance().log("Client: Finished RByz\n");
}