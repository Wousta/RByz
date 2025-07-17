#include "rbyzAux.hpp"
#include "attacks.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"

#include <ATen/ops/flatten.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

# define CLNT_FINISHED_RBYZ -14

//////////////////////////////////////////////////////////////
////////////////////// SERVER FUNCTIONS //////////////////////
void RByzAux::awaitTermination(std::vector<ClientDataRbyz>& clnt_data_vec, int code) {
  Logger::instance().log("Server: Waiting for all clients to complete Task...\n");
  for (int i = 0; i < clnt_data_vec.size(); i++) {
    if (clnt_data_vec[i].is_byzantine) {
      continue;
    }

    Logger::instance().log("Server: Waiting for client " + std::to_string(i) + 
                            " to finish RByz round " + std::to_string(code) + "\n");

    bool client_done = false;
    while (!client_done) {
      // Check if client has reached final round
      if (clnt_data_vec[i].round == code) {
        client_done = true;
        Logger::instance().log("Server: Client " + std::to_string(i) + " has finished\n");
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::instance().log("Client " + std::to_string(i) + " in round " + std::to_string(clnt_data_vec[i].round) + "\n");
      }
    }
  }
}

void RByzAux::renewTrustedClientsColumn(RegMnistSplitter &splitter, std::vector<ClientDataRbyz> &clnt_data_vec) {
  int clnts_to_renew = 0;
  for (auto& client : clnt_data_vec) {
    if (client.trust_score <= ts_threshold) {
      continue;
    }

    int samples_per_section = splitter.getExtraColNumSamplesPerSection();
    size_t section_idx = splitter.getExtraColSectionToRenew();
    Logger::instance().log("  Renewing from client " + std::to_string(client.index) + 
                           " at section " + std::to_string(section_idx) + "\n");

    std::vector<uint32_t> indices_restore;
    indices_restore.reserve(samples_per_section);
    Logger::instance().log("    Images original indices to renew:");
    for (int i = 0; i < samples_per_section; i++) {
      size_t image_idx = section_idx + i;
      indices_restore.push_back(*mngr.getOriginalIndex(image_idx));
      if (i % 16 == 0)
        Logger::instance().log("\n    | ");
      
      Logger::instance().log(std::to_string(*mngr.getOriginalIndex(image_idx)) + " ");
    }

    int possible_clnt_indices = client.num_samples - samples_per_section;

    // Idk why but trying to read/write from the last registered sample causes issues, I checked sizes
    possible_clnt_indices -= 1; 
    std::uniform_int_distribution<int> dist(0, possible_clnt_indices);
    int clnt_idx = dist(rng);
    uint64_t remote_offset = clnt_idx * mngr.data_info.get_sample_size();
    uint64_t local_offset = mngr.getSampleOffset(section_idx);
    size_t rdma_size = splitter.getExtraColumnSize();

    LocalInfo local_info;
    local_info.indices.push_back(REG_DATASET_IDX);
    local_info.offs.push_back(local_offset);
    RemoteInfo remote_info;
    remote_info.indx = REG_DATASET_IDX;
    remote_info.off = remote_offset;
    rdma_ops.exec_rdma_read(rdma_size, local_info, remote_info, client.index, false);

    Logger::instance().log("\n    Images original indices pre renew:");
    for (int i = 0; i < samples_per_section; i++) {
      size_t image_idx = section_idx + i;
      if (i % 16 == 0)
        Logger::instance().log("\n    ] ");
      Logger::instance().log(std::to_string(*mngr.getOriginalIndex(image_idx)) + " ");
    }

    // Restore original indices after RDMA read
    for (int i = 0; i < samples_per_section; i++) {
      size_t image_idx = section_idx + i;
      *mngr.getOriginalIndex(image_idx) = indices_restore[i];
    }

    // Log the renewed labels
    Logger::instance().log("\n    Images Indices post renew:");
    for (int i = 0; i < samples_per_section; i++) {
      size_t image_idx = section_idx + i;
      if (i % 16 == 0)
        Logger::instance().log("\n    + ");
      Logger::instance().log(std::to_string(*mngr.getOriginalIndex(image_idx)) + " ");
    }
    Logger::instance().log("\n");

    clnts_to_renew++;
  }
  Logger::instance().log("==========Renewed from " + std::to_string(clnts_to_renew) + 
                         " clients' extra column samples\n");

  Logger::instance().logCustom(t_params.logs_dir, 
      t_params.clnts_renew_file, 
      std::to_string(clnts_to_renew) + "\n");
}

void RByzAux::updateTS(std::vector<ClientDataRbyz> &clnt_data_vec,
                       ClientDataRbyz &clnt_data, float srvr_loss,
                       float srvr_error_rate) {
  // loss and error of client to update
  float w_loss = clnt_data.loss;
  float w_err = clnt_data.error_rate;

  // Logger::instance().log("        -> Client " + std::to_string(clnt_data.index) + 
  //                        " loss: " + std::to_string(w_loss) + 
  //                        ", error rate: " + std::to_string(w_err) + "\n");

  // Find minimum loss and error among all clients
  float min_w_loss = w_loss;
  float min_w_err = w_err;
  for (const ClientDataRbyz &clnt_data : clnt_data_vec) {
    if (clnt_data.loss < min_w_loss) {
      min_w_loss = clnt_data.loss;
    }

    if (clnt_data.error_rate < min_w_err) {
      min_w_err = clnt_data.error_rate;
    }
  }

  // Logger::instance().log("        -> Minimum loss: " + std::to_string(min_w_loss) + 
  //                        ", Minimum error rate: " + std::to_string(min_w_err) + "\n");

  float no_bias = 0.001; // Tweak until satisfied
  float max_loss = std::max(0.0f, srvr_loss - w_loss);
  float loss_Calc = max_loss / (srvr_loss - min_w_loss + no_bias);
  float max_err = std::max(0.0f, srvr_error_rate - w_err);
  float err_Calc = max_err / (srvr_error_rate - min_w_err + no_bias);

  // Logger::instance().log("        -> Server loss: " + std::to_string(srvr_loss) + 
  //                        ", Server error rate: " + std::to_string(srvr_error_rate) + "\n");
  // Logger::instance().log("        -> Max loss: " + std::to_string(max_loss) + 
  //                        ", Max error rate: " + std::to_string(max_err) + "\n");
  // Logger::instance().log("        -> Calculated loss_Calc: " + std::to_string(loss_Calc) + 
  //                        ", err_Calc: " + std::to_string(err_Calc) + "\n");

  clnt_data.trust_score = (loss_Calc + err_Calc) / 2;
}

torch::Tensor
RByzAux::aggregate_updates(const std::vector<torch::Tensor> &client_updates,
                           const std::vector<ClientDataRbyz> &clnt_data_vec,
                           const std::vector<uint32_t> &clnt_indices) {
  Logger::instance().log("\nComputing aggregation data ================\n");
  float trust_scores[client_updates.size()];
  for (int i = 0; i < client_updates.size(); i++) {
    // Byzantine clients are skipped, so we have to keep track of good clients indices clnt_indices[i]
    trust_scores[i] = clnt_data_vec[clnt_indices[i]].trust_score;
  }

  float sum_trust = 0.0f;
  for (float score : trust_scores) {
    sum_trust += score;
  }

  torch::Tensor aggregated_update = torch::zeros_like(client_updates[0]);
  if (sum_trust > 0) {
    for (int i = 0; i < client_updates.size(); i++) {
      aggregated_update += client_updates[i] * trust_scores[i];
    }

    aggregated_update /= sum_trust;
  }

  Logger::instance().log("Final Aggregated update: " + aggregated_update.slice(0, 0, std::min<size_t>(aggregated_update.numel(), 5)).toString() + "\n");
  Logger::instance().log("================================\n");

  return aggregated_update;
}

void RByzAux::writeServerVD(RegMnistSplitter &splitter,
                            std::vector<ClientDataRbyz> &clnt_data_vec,
                            float proportion) {

  std::vector<int> derangement = splitter.generateDerangement();

  // Send the VD samples to the clients
  for (int i = 0; i < clnt_data_vec.size(); i++) {
    if (clnt_data_vec[i].is_byzantine) {
      Logger::instance().log("Server: Client " + std::to_string(i) +
                             " is Byzantine, skipping VD sample sending\n");
      continue;
    }

    std::vector<size_t> srvr_indices =
        splitter.getServerIndices(i, derangement);
    std::vector<size_t> clnt_chunks =
        splitter.getClientChunks(clnt_data_vec[i].index, proportion);

    size_t srvr_idx = 0;
    for (size_t j = 0; j < clnt_chunks.size(); j++) {

      size_t clnt_chunk = clnt_chunks[j];

      LocalInfo local_info;
      local_info.indices.push_back(REG_DATASET_IDX);
      RemoteInfo remote_info;
      remote_info.indx = REG_DATASET_IDX;
      remote_info.off = clnt_chunk;

      if (srvr_idx >= srvr_indices.size()) {
        Logger::instance().log("Warning: Not enough VD samples for client " + std::to_string(i) +
                              "srvr_idx: " + std::to_string(srvr_idx) +
                              " srvr_indices.size(): " + std::to_string(srvr_indices.size()) + "\n");
        break;
      }

      size_t srvr_sample_idx = srvr_indices[srvr_idx++];
      uint64_t sample_offset = mngr.getSampleOffset(srvr_sample_idx);
      local_info.offs.push_back(sample_offset);
      rdma_ops.exec_rdma_write(splitter.getChunkSize(), local_info, remote_info,
      i, false);

      for (int k = 0; k < splitter.getSamplesPerChunk(); k++) {
        clnt_data_vec[i].inserted_indices.insert(srvr_sample_idx++);
      }
    }
  }
  Logger::instance().log("Server: VD samples sent to all clients\n");
}

bool RByzAux::processVDOut(ClientDataRbyz &clnt_data, bool check_byz) {
  // Compare the output of the forward pass with the server's output
  float* clnt_out = clnt_data.forward_pass;
  uint32_t* clnt_indices = clnt_data.forward_pass_indices;
  float* srvr_out = mngr.f_pass_data.forward_pass;
  uint32_t* srvr_indices = mngr.f_pass_data.forward_pass_indices;
  size_t clnt_f_pass_size = clnt_data.forward_pass_mem_size / sizeof(float);
  
  size_t clnt_num_samples = clnt_data.forward_pass_indices_mem_size / sizeof(uint32_t);
  size_t srvr_num_samples = mngr.f_pass_data.forward_pass_indices_mem_size / sizeof(uint32_t);
  size_t clnt_error_start = clnt_f_pass_size / 2;

  // Calculate server error start position
  size_t srvr_f_pass_size = mngr.f_pass_data.forward_pass_mem_size / sizeof(float);
  size_t srvr_error_start = srvr_f_pass_size / 2;

  // Divided by two because forward pass output contains both loss and error
  // rate
  if (clnt_f_pass_size / 2 != clnt_num_samples) {
    throw std::runtime_error("[processVDOut] Forward pass and indices sizes do not match");
  } 

  Logger::instance().log("$$$$$$$ processVDOut client " + std::to_string(clnt_data.index) + " $$$$$$$\n");
  // get the original indices of the inserted samples
  std::unordered_set<uint32_t> inserted_indices_set;
  // Logger::instance().log("  > Inserted indices:");
  int logi = 0;
  for (size_t srvr_idx : clnt_data.inserted_indices) {
    // if (logi++ % 10 == 0)
    //   Logger::instance().log("\n    $ ");
    // Logger::instance().log(std::to_string(srvr_idx) + " ");
    if (inserted_indices_set.find(*mngr.getOriginalIndex(srvr_idx)) != inserted_indices_set.end()) {
      Logger::instance().log("Warning: Duplicate index found: server index " + std::to_string(srvr_idx) + 
                             " original index " + std::to_string(*mngr.getOriginalIndex(srvr_idx)) + "\n");
    }
    inserted_indices_set.insert(*mngr.getOriginalIndex(srvr_idx));
  }

  if (inserted_indices_set.size() != clnt_data.inserted_indices.size()) {
    throw std::runtime_error("[processVDOut] Inserted indices set size does not match the number of inserted indices " +
                              std::to_string(clnt_data.inserted_indices.size()) + 
                              " vs " + std::to_string(inserted_indices_set.size()));
  }

  Logger::instance().log("  > Inserted indices for client " + std::to_string(clnt_data.index) + 
                         ": " + std::to_string(inserted_indices_set.size()) + 
                         " indices\n");

  // Map original indices of forward pass to their positions in the server output
  std::unordered_map<uint32_t, int> srvr_indices_map;
  for (size_t i = 0; i < srvr_num_samples; ++i) {
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
  Logger::instance().log("  > Processing " + std::to_string(clnt_num_samples) + 
                         " samples for client " + std::to_string(clnt_data.index));
  // Check only the inserted indices (VD samples) to minimize comparisons
  for (size_t i = 0; i < clnt_num_samples && validation_passed; ++i) {
    uint32_t clnt_idx = clnt_indices[i];
    // if (i % 10 == 0) {
    //   Logger::instance().log("\n    | ");
    // }
    // Logger::instance().log(std::to_string(clnt_idx) + " ");

    // Only compare samples that were inserted by the server
    if (inserted_indices_set.find(clnt_idx) != inserted_indices_set.end()) {
      // Find corresponding server index
      auto srvr_it = srvr_indices_map.find(clnt_idx);
      if (srvr_it == srvr_indices_map.end()) {
        throw std::runtime_error("Missing server index for inserted sample: " +
                                 std::to_string(clnt_idx));
      }

      size_t srvr_i = srvr_it->second;

      clnt_loss_total += clnt_out[i];
      srvr_loss_total += srvr_out[srvr_i];
      clnt_error_total += clnt_out[clnt_error_start + i];
      srvr_error_total += srvr_out[srvr_error_start + srvr_i];

      if (processed_samples < 1) {
        Logger::instance().log("    > Processing sample of client" + std::to_string(clnt_data.index) + 
                              ": idx " + std::to_string(clnt_idx) + 
                              " loss: " + std::to_string(clnt_out[i]) + 
                              " err: " + std::to_string(clnt_out[clnt_error_start + i]) + 
                              " | Server loss: " + std::to_string(srvr_out[srvr_i]) +
                              " Server err: " + std::to_string(srvr_out[srvr_error_start + srvr_i]) + "\n");
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
                                " | Err: client=" + std::to_string(clnt_error) + 
                                ", server=" + std::to_string(srvr_error) + "\n");
          validation_passed = false;
        } else {
          Logger::instance().log("!Match at index " + std::to_string(clnt_idx) + 
                                " - Loss: client=" + std::to_string(clnt_loss) +
                                ", server=" + std::to_string(srvr_loss) +
                                " | Err: client=" + std::to_string(clnt_error) +
                                ", server=" + std::to_string(srvr_error) + "\n");
        }
      }
    }
  }

  clnt_data.loss = clnt_loss_total / inserted_indices_set.size();
  clnt_data.error_rate = clnt_error_total / inserted_indices_set.size();
  clnt_data.loss_srvr = srvr_loss_total / inserted_indices_set.size();
  clnt_data.error_rate_srvr = srvr_error_total / inserted_indices_set.size();

  float processed_samples_proportion =
      static_cast<float>(processed_samples) / clnt_num_samples;
  if (processed_samples_proportion < 0.01) {
    Logger::instance().log("    WARNING processVDOut: Client " + std::to_string(clnt_data.index) + 
                             " Processed samples below 1%: " +
                             std::to_string(processed_samples) + " vs " + 
                             std::to_string(clnt_num_samples) + "\n");

    validation_passed = false;
  }

  Logger::instance().log("    Client " + std::to_string(clnt_data.index) + 
                         " VD proc results - Loss: " + std::to_string(clnt_data.loss) + 
                         ", Error Rate: " + std::to_string(clnt_data.error_rate) + 
                          " | Server Loss: " + std::to_string(clnt_data.loss_srvr) +
                          ", Server Err_Rate: " + std::to_string(clnt_data.error_rate_srvr) + "\n");
  
  // All comparisons passed
  Logger::instance().log("    $ VD validation passed for a proportion of " +
                         std::to_string(processed_samples_proportion * 100) + "%' inserted samples" +
                         "for client " + std::to_string(clnt_data.index) + " ·····$\n");
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

void RByzAux::logTrustScores(const std::vector<ClientDataRbyz> &clnt_data_vec,
                             int only_flt) const {
  std::string filename = t_params.ts_file;
  Logger::instance().logCustom(t_params.logs_dir, filename, "- Round end -\n");

  for (int i = 0; i < clnt_data_vec.size(); i++) {
    std::string message = std::to_string(clnt_data_vec[i].trust_score) + "\n";
    Logger::instance().logCustom(t_params.logs_dir, filename, message);
  }
}

void RByzAux::waitInfinite(ClientDataRbyz &clnt_data, int round) {
  int clnt_idx = clnt_data.index;

  // Wait for client to finish the step indefinitely
  long total_time_waited = 0;
  std::chrono::milliseconds initial_time(static_cast<long>(clnt_data.limit_step_time.count()));
  // Logger::instance().log("    Waiting for client: " +  std::to_string(clnt_idx) + 
  //                        " to finish step " + std::to_string(clnt_data.local_step) + 
  //                        ", round: " + std::to_string(clnt_data.round) + "\n");

  // Logger::instance().log("     -> PRE: Client local step: " + std::to_string(clnt_data.local_step) + 
  //                        ", next step: " + std::to_string(clnt_data.next_step) + 
  //                        ", round: " + std::to_string(clnt_data.round) + 
  //                         ", round to wait: " + std::to_string(round) + "\n");

  if (clnt_data.local_step < clnt_data.next_step && clnt_data.round <= round) {
    std::this_thread::sleep_for(initial_time);
    total_time_waited += initial_time.count();
    Logger::instance().log("     -> Server waited initial: " + std::to_string(total_time_waited) +
                            " ms for client " + std::to_string(clnt_idx) + "\n");
  }

  rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, clnt_idx);
  rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, clnt_idx);

  std::chrono::milliseconds exp_backoff_time(1);
  bool advanced = true;

  while (clnt_data.local_step < clnt_data.next_step &&
         clnt_data.round <= round) {
    std::this_thread::sleep_for(exp_backoff_time);
    total_time_waited += exp_backoff_time.count();
    exp_backoff_time =
        std::chrono::milliseconds(exp_backoff_time.count() * 3 / 2);
    rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, clnt_idx);
    rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, clnt_idx);
  }

  // Logger::instance().log("     -> POST Client local step: " + std::to_string(clnt_data.local_step) + 
  //                       ", next step: " + std::to_string(clnt_data.next_step) + 
  //                       ", round: " + std::to_string(clnt_data.round) + 
  //                       ", round to wait: " + std::to_string(round) + "\n");

  clnt_data.next_step += 1;
  if (total_time_waited > 0)
    Logger::instance().log("     -> Server waited: " + std::to_string(total_time_waited) + " ms\n");
}

void RByzAux::waitTimeout(ClientDataRbyz &clnt_data, int round) {
  int clnt_idx = clnt_data.index;

  // Wait for client to finish the step with exponential backoff
  long total_time_waited = 0;
  std::chrono::milliseconds initial_time(static_cast<long>(clnt_data.limit_step_time.count()));

  if (clnt_data.local_step < clnt_data.next_step && clnt_data.round <= round) {
    std::this_thread::sleep_for(initial_time);
    total_time_waited += initial_time.count();
    Logger::instance().log("    -> Server waited initial: " + std::to_string(total_time_waited) +
                            " ms for client " + std::to_string(clnt_idx) + "\n");
  }

  rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, clnt_idx);
  rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, clnt_idx);

  // Logger::instance().log("    -> Client local step: " + std::to_string(clnt_data.local_step) + 
  //                        ", next step: " + std::to_string(clnt_data.next_step) + 
  //                        ", round: " + std::to_string(clnt_data.round) + 
  //                         ", round to wait: " + std::to_string(round) + "\n");
  std::chrono::milliseconds exp_backoff_time(1);
  bool timed_out = false;
  while (!timed_out && clnt_data.local_step < clnt_data.next_step && clnt_data.round <= round) {
    std::this_thread::sleep_for(exp_backoff_time);
    total_time_waited += exp_backoff_time.count();
    exp_backoff_time =
        std::chrono::milliseconds(exp_backoff_time.count() * 3 / 2);
    if (exp_backoff_time.count() > clnt_data.limit_step_time.count() * 0.2) {

      // Choose lower steps to wait for and increase the limit time
      if(clnt_data.steps_to_finish <= min_steps) {
        Logger::instance().log("    -> Server: Client " + std::to_string(clnt_idx) + 
                                " is Byzantine, skipping\n");
        clnt_data.is_byzantine = true;

      } else {
        long int new_limit = static_cast<long int>(std::ceil(clnt_data.limit_step_time.count() * 1.3));
        clnt_data.limit_step_time = std::chrono::milliseconds(new_limit);
        clnt_data.steps_to_finish = step_range(rng);
        middle_steps = std::max(static_cast<int>(std::floor(clnt_data.steps_to_finish * 0.75)), min_steps);
        step_range = std::uniform_int_distribution<int>(middle_steps, clnt_data.steps_to_finish);
        Logger::instance().log("    -> Client " + std::to_string(clnt_idx) + 
                                " lowering steps to finish to: " + std::to_string(clnt_data.steps_to_finish) + "\n");

        // Client will now do steps_to_finish local steps
        clnt_data.clnt_CAS.store(clnt_data.steps_to_finish);
        rdma_ops.exec_rdma_write(sizeof(int), CLNT_CAS_IDX, clnt_idx);
      }

      Logger::instance().log("    -> Server waiting: Client " + std::to_string(clnt_idx) + " timed out\n");
      timed_out = true;
    }

    rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, clnt_idx);
    rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, clnt_idx);
  }
  Logger::instance().log("    -> Server waited: " + std::to_string(total_time_waited) + " ms\n");

  if (!timed_out) {
    clnt_data.next_step = clnt_data.local_step + 1;
  }
}

//////////////////////////////////////////////////////////////
/////////////////////// RBYZ ALGORITHM ///////////////////////
void RByzAux::runRByzServer(int n_clients, std::vector<torch::Tensor> &w,
                            RegMemSrvr &regMem,
                            std::vector<ClientDataRbyz> &clnt_data_vec) {

  Logger::instance().logCustom(t_params.logs_dir, t_params.clnts_renew_file,
                               "TS_threshold " + std::to_string(ts_threshold) + "\n");

  Logger::instance().log("\n\n==============  STARTING RBYZ  ==============\n");
  Logger::instance().log("reg_sz_data = " +  std::to_string(regMem.reg_sz_data) + "\n"); 
  int total_steps = t_params.global_iters_fl;
  for (int &clnt_flag  : regMem.clnt_ready_flags) {
    clnt_flag = 0; // Reset client ready flags
  }

  // Set manager epochs to 1, the epochs will be controled by RByz
  mngr.kNumberOfEpochs = 1;

  // Create VD splits and do first write of VD to the clients
  RegMnistSplitter splitter(t_params, mngr, clnt_data_vec);

  // RBYZ training loop
  for (int round = 1; round <= global_rounds; round++) {
    Logger::instance().log("\n\n=================  ROUND " + std::to_string(round) + " STARTED  =================\n");

    // Write the current model weights to the server registered memory 
    tops::memcpyTensorVec(regMem.srvr_w, w, regMem.reg_sz_data);

    // Before each round, write the server's VD to the clients to test after first local step
    if (round == 1) {
      initTimeoutTime(clnt_data_vec);
      writeServerVD(splitter, clnt_data_vec, t_params.vd_prop_write);
    }

    if (round % t_params.test_renewal_freq == 0 && round > 3) {
      Logger::instance().log("Writing test samples for round " + std::to_string(round) + "\n");
      Logger::instance().log("Renewing dataset for round " + std::to_string(round) + "\n");
      renewTrustedClientsColumn(splitter, clnt_data_vec);
      writeServerVD(splitter, clnt_data_vec, t_params.vd_prop_write);
    }

    // Signal to clients that the server is ready
    regMem.srvr_ready_rb.store(round);

    // For each client run N rounds of RByz
    for (int srvr_step = 0; srvr_step < local_steps; srvr_step++) {
      Logger::instance().log(" -- Server: Running step " + std::to_string(srvr_step) + " of RByz\n");

      for (ClientDataRbyz &clnt_data : clnt_data_vec) {
        rdma_ops.exec_rdma_read(sizeof(int), CLNT_LOCAL_STEP_IDX, clnt_data.index);
        rdma_ops.exec_rdma_read(sizeof(int), CLNT_ROUND_IDX, clnt_data.index);

        // Skip clients whose steps were lowered or are Byzantine
        if (clnt_data.is_byzantine) {
          Logger::instance().log("  WARNING: skipping Client " + std::to_string(clnt_data.index) + "\n");
          continue;
        }

        if (t_params.wait_all) {
          waitInfinite(clnt_data, round);
        } else {
          waitTimeout(clnt_data, round);
        }

        Logger::instance().log("    -> Client " + std::to_string(clnt_data.index) + 
                        " local step = " + std::to_string(clnt_data.local_step) + 
                        " server step = " + std::to_string(srvr_step) + 
                        " dataset size = " + std::to_string(clnt_data.dataset_size) + "\n");
      }
    }

    // Run inference on the server before comparing with clients
    mngr.runInference();

    int included = 0;
    for (ClientDataRbyz &clnt_data : clnt_data_vec) {
      if (clnt_data.is_byzantine) {
        continue;
      }
      rdma_ops.exec_rdma_read(clnt_data.forward_pass_mem_size,
                              CLNT_FORWARD_PASS_IDX, clnt_data.index);
      rdma_ops.exec_rdma_read(clnt_data.forward_pass_indices_mem_size,
                              CLNT_FORWARD_PASS_INDICES_IDX, clnt_data.index);

      // Only update TS if at least 50% of the server's VD samples were processed
      clnt_data.include_in_agg = processVDOut(clnt_data, false);
      if (clnt_data.include_in_agg) {
        updateTS(clnt_data_vec, clnt_data, clnt_data.loss_srvr,
          clnt_data.error_rate_srvr);

        if (clnt_data.trust_score != 0) {
          included++;
        }
      } 

      // Reset next step counter for the next round
      clnt_data.next_step = 1;
    }
    Logger::instance().logCustom(t_params.logs_dir, t_params.included_agg_file,
                                 std::to_string(included) + "\n");

    Logger::instance().log("    Trust Scores after last step of round " + std::to_string(round) + ":\n");
    for (const ClientDataRbyz& clnt_data : clnt_data_vec) {
      Logger::instance().log("    Client " + std::to_string(clnt_data.index) +
                              " Trust Score: " + std::to_string(clnt_data.trust_score) + "\n");
    }

    // For the experiments
    logTrustScores(clnt_data_vec, t_params.only_flt);

    std::vector<torch::Tensor> clnt_updates;
    std::vector<uint32_t> clnt_indices;

    // Read client updates and aggregate them
    for (size_t i = 0; i < clnt_data_vec.size(); i++) {
      ClientDataRbyz &client = clnt_data_vec[i];

      if (client.is_byzantine) {
        Logger::instance().log("    -> Client " + std::to_string(client.index) +
                               " is Byzantine, skipping\n");
        continue;
      }

      bool timed_out = false;
      long total_time_waited = 0;
      std::chrono::microseconds initial_time(20); // time of 10 round trips
      std::chrono::microseconds limit_step_time(30000000); // 30 seconds
      Logger::instance().log("    Waiting for client " + std::to_string(client.index) +
                             " to finish round " + std::to_string(round) + "\n");

      // Wait for the client to finish the round
      while (regMem.clnt_ready_flags[client.index] != round && !timed_out) {
        std::this_thread::sleep_for(initial_time);
        total_time_waited += initial_time.count();
        initial_time = std::chrono::microseconds(
            initial_time.count() * 3 / 2); // Exponential backoff
        if (initial_time > limit_step_time) {
          timed_out = true;
          Logger::instance().log(
              "    -> Timeout in update gathering by client " +
              std::to_string(client.index) + " for round " +
              std::to_string(round) +
              " client round: " + std::to_string(client.round) + "\n");
        }
      }

      if (total_time_waited > 0)
        Logger::instance().log(
            "    -> Server waited: " + std::to_string(total_time_waited) +
            " us for client " + std::to_string(client.index) + "\n");

      if (!timed_out && client.trust_score != 0 && client.include_in_agg) {
        size_t numel_server = regMem.reg_sz_data / sizeof(float);
        torch::Tensor flat_tensor =
            torch::from_blob(regMem.clnt_ws[i],
                             {static_cast<long>(numel_server)}, torch::kFloat32)
                .clone();

        clnt_updates.push_back(flat_tensor);
        clnt_indices.push_back(client.index);
      }
    }

    torch::Tensor aggregated_update;
    if (clnt_updates.empty()) {
      torch::Tensor w_flat = tops::flatten_tensor_vector(w);
      aggregated_update = torch::zeros_like(w_flat);
    } else {
      aggregated_update = aggregate_updates(clnt_updates, clnt_data_vec, clnt_indices);
    }

    std::vector<torch::Tensor> aggregated_update_vec =
        tops::reconstruct_tensor_vector(aggregated_update, w);

    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] + t_params.global_learn_rate * aggregated_update_vec[i];
    }
    Logger::instance().log("After aggregating round " + std::to_string(round) + ":\n");
    mngr.updateModelParameters(w);
    mngr.runTesting();

    // Log accuracy and round to Results
    total_steps++;
    Logger::instance().logCustom(t_params.logs_dir, t_params.acc_file,
                                 std::to_string(total_steps) + " " +
                                     std::to_string(mngr.test_accuracy) + "\n");

    std::cout << "///////////////// Server: Round " << round << " completed /////////////////\n";
    Logger::instance().log("\n//////////////// Server: Round " + std::to_string(round) + " completed ////////////////\n");
  }

  Logger::instance().log("\n\n=================  RBYZ SERVER FINISHED  =================\n");
  std::cout << "\n\n=================  RBYZ SERVER FINISHED  =================\n";
  awaitTermination(clnt_data_vec, CLNT_FINISHED_RBYZ);  
}

/**
 * @brief Run the RByz client, only the clients call this function.
 */
void RByzAux::runRByzClient(std::vector<torch::Tensor> &w, RegMemClnt &regMem) {
  Logger::instance().log("\n\n==============  STARTING RBYZ  ==============\n");
  Logger::instance().log("Client: Starting RByz with accuracy\n");
  // std::string log_file = "stepTimes_" + std::to_string(regMem.id) + ".log";
  // Set manager epochs to 1, the epochs will be controled by RByz
  mngr.kNumberOfEpochs = 1;
  regMem.clnt_ready_flag.store(0);

  while (regMem.round <= global_rounds) {
    if (regMem.round == 2) {
      Logger::instance().log("POST: first mnist samples\n");
      for (int i = 0; i < 320; i++) {
        if (i % 32 == 0)
          Logger::instance().log(
              "Sample " + std::to_string(i) +
              ": label = " + std::to_string(*mngr.getLabel(i)) +
              " | og_idx = " + std::to_string(*mngr.getOriginalIndex(i)) + "\n");
      }
    }

    // Reset
    regMem.local_step.store(0);

    Logger::instance().log("\n//////////////// Round " + std::to_string(regMem.round) + " started ////////////////\n");
    // Wait for the server to be ready
    do {
      rdma_ops.exec_rdma_read(sizeof(int), SRVR_READY_RB_IDX);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (regMem.srvr_ready_rb < regMem.round && regMem.srvr_ready_rb != SRVR_FINISHED);

    if (regMem.srvr_ready_rb == SRVR_FINISHED) {
      Logger::instance().log("Server finished early, exiting...\n");
      return;
    }

    if (regMem.round < regMem.srvr_ready_rb) {
      regMem.round.store(regMem.srvr_ready_rb);
    }

    rdma_ops.exec_rdma_read(regMem.reg_sz_data, SRVR_W_IDX);
    tops::writeToTensorVec(w, regMem.srvr_w, regMem.reg_sz_data);
    Logger::instance().log("Round " + std::to_string(regMem.round.load()) +
                           " weights received:\n");

    std::vector<torch::Tensor> w_pre_train = mngr.updateModelParameters(w);

    Logger::instance().log(
        "Steps to run: " + std::to_string(regMem.CAS.load()) + "\n");
    while (regMem.local_step.load() < regMem.CAS.load()) {
      // auto start = std::chrono::high_resolution_clock::now();
      int step = regMem.local_step.load();
      Logger::instance().log(" ...... Client: Running step " +
                             std::to_string(step) + " of RByz in round " +
                             std::to_string(regMem.round.load()) + "\n");
      mngr.runTraining();

      regMem.local_step.store(step + 1);

      // auto end = std::chrono::high_resolution_clock::now();
      // std::string time = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
      // Logger::instance().logCustom("./stepTimes", log_file, time + "\n");
      // Logger::instance().log("Client: Local step " + std::to_string(regMem.local_step.load()) + " going ahead\n");
    }

    // Write the updates to the registered memory and write to the server
    std::vector<torch::Tensor> g = mngr.calculateUpdate(w_pre_train);
    tops::memcpyTensorVec(regMem.clnt_w, g, regMem.reg_sz_data);
    rdma_ops.exec_rdma_write(regMem.reg_sz_data, CLNT_W_IDX);

    // If overwriting poisoned labels is enabled, byz client has to renew
    // poisoned labels (50% chance)
    if (byz_clnt && t_params.overwrite_poisoned && coinFlip()) {
      data_poison_attack(t_params.use_mnist, t_params, mngr);
    }

    regMem.clnt_ready_flag.store(regMem.round);
    rdma_ops.exec_rdma_write(sizeof(int), CLNT_READY_IDX);
    regMem.round.store(regMem.round + 1);

    Logger::instance().log("\n//////////////// Client: Round " +
                           std::to_string(regMem.round - 1) +
                           " completed ////////////////\n");
  }

  // Notify the server that the client is done
  regMem.round.store(CLNT_FINISHED_RBYZ);
  rdma_ops.exec_rdma_write(sizeof(int), CLNT_ROUND_IDX);
  Logger::instance().log("Finish round " + std::to_string(regMem.round.load()) + 
                         " and notify server\n");
  Logger::instance().log("Client: Finished RByz\n");
}