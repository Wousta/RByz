#include "fltrustSrvr.hpp"

#include "tensorOps.hpp"

std::vector<int> FLtrustSrvr::generateRandomUniqueVector(int n_clients, int min_sz) {
  if (min_sz == -1) {
    min_sz = n_clients;
  }

  std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
  std::vector<int> allValues(n_clients);
  for (int i = 0; i < n_clients; i++) {
    allValues[i] = i;
  }
  std::shuffle(allValues.begin(), allValues.end(), rng);
  std::uniform_int_distribution<int> sizeDist(min_sz, n_clients);

  int size = sizeDist(rng);
  std::vector<int> result(allValues.begin(), allValues.begin() + size);
  std::sort(result.begin(), result.end());

  return result;
}

torch::Tensor FLtrustSrvr::aggregateUpdates(const std::vector<torch::Tensor> &clnt_updates,
                                            const torch::Tensor &server_update,
                                            const std::vector<uint32_t> &clnt_indices) {
  if (clnt_updates.empty()) {
    return server_update.clone();
  }

  std::vector<float> trust_scores;
  std::vector<torch::Tensor> norm_updates;
  trust_scores.reserve(clnt_updates.size());
  norm_updates.reserve(clnt_updates.size());

  float server_norm = torch::norm(server_update, 2).item<float>();
  for (int i = 0; i < clnt_updates.size(); i++) {
    // Compute cosine similarityç
    torch::Tensor dot_product = torch::dot(clnt_updates[i], server_update);
    float client_norm = torch::norm(clnt_updates[i], 2).item<float>();
    float cosine_sim = dot_product.item<float>() / (client_norm * server_norm);

    // Apply ReLU
    float trust_score = std::max(0.0f, cosine_sim);
    trust_scores.push_back(trust_score);

    // Normalize
    torch::Tensor norm_update = clnt_updates[i] * (server_norm / client_norm);
    norm_updates.push_back(norm_update);
  }

  Logger::instance().logCustom(t_params.logs_dir, t_params.ts_file, "- Round end -\n");
  for (int i = 0; i < log_TS_vec.size(); i++) {
    std::string message = std::to_string(log_TS_vec[i]) + "\n";
    Logger::instance().logCustom(t_params.logs_dir, t_params.ts_file, message);
  }

  float sum_trust = 0.0f;
  for (float score : trust_scores) {
    sum_trust += score;
  }

  torch::Tensor aggr_update = torch::zeros_like(server_update);
  if (sum_trust > 0) {
    for (int i = 0; i < trust_scores.size(); i++) {
      aggr_update += trust_scores[i] * norm_updates[i];
    }
    aggr_update /= sum_trust;
  }

  return aggr_update;
}

bool FLtrustSrvr::expBackoffWait(int round, int client) {
  bool timed_out = false;
  long total_wait_time = 0;
  std::chrono::milliseconds initial_time(1);
  std::chrono::milliseconds limit_step_time(30000);

  while (regMem.clnt_ready_flags[client] != round && !timed_out) {
    std::this_thread::sleep_for(initial_time);
    total_wait_time += initial_time.count();
    int64_t new_time = initial_time.count() * 3 / 2;
    initial_time = std::chrono::milliseconds(new_time);

    if (initial_time > limit_step_time) {
      timed_out = true;
    }
  }

  return timed_out;
}

std::vector<torch::Tensor> FLtrustSrvr::run() {
  std::vector<torch::Tensor> w = mngr.getInitialWeights();

  Logger::instance().log("\nInitial weights gathered\n");
  std::vector<float> log_TS_vec(t_params.n_clients, 0.0f);

  for (int round = 1; round <= t_params.global_iters_fl; round++) {
    std::vector<int> polled_clients = generateRandomUniqueVector(t_params.n_clients, -1);
    tops::memcpyTensorVec(regMem.srvr_w, w, regMem.reg_sz_data);

    // Set the flag to indicate that the weights are ready for the clients to
    // read
    regMem.srvr_ready_flag.store(round);

    // Run local training
    std::vector<torch::Tensor> w_pre_train = mngr.updateModelParameters(w);
    mngr.runTraining();

    // g is the delta
    std::vector<torch::Tensor> g = mngr.calculateUpdate(w_pre_train);

    std::vector<torch::Tensor> clnt_updates;
    std::vector<uint32_t> clnt_indices;
    clnt_updates.reserve(polled_clients.size());
    clnt_indices.reserve(polled_clients.size());
    Logger::instance().log("polled_clients size: " + std::to_string(polled_clients.size()) + ":\n");

    for (size_t i = 0; i < polled_clients.size(); i++) {
      int client = polled_clients[i];
      if (!expBackoffWait(round, client)) {
        size_t numel_server = regMem.reg_sz_data / sizeof(float);
        torch::Tensor flat_tensor =
            torch::from_blob(regMem.clnt_ws[client], {static_cast<long>(numel_server)},
                             torch::kFloat32)
                .clone();

        torch::Tensor learn_params = mngr.extractLearnableParams(flat_tensor);
        clnt_updates.push_back(learn_params);
        clnt_indices.push_back(client);
      }
    }

    // AGGREGATION PHASE //////////////////////
    torch::Tensor srvr_full_update = tops::flatten_tensor_vector(g);
    torch::Tensor srvr_update = mngr.extractLearnableParams(srvr_full_update);
    torch::Tensor aggregated_update = aggregateUpdates(clnt_updates, srvr_update, clnt_indices);
    std::vector<torch::Tensor> aggregated_update_vec =
        tops::reconstruct_tensor_vector(aggregated_update, w);

    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] + t_params.global_learn_rate * aggregated_update_vec[i];
    }
    mngr.updateModelParameters(w);
    mngr.runTesting();
    Logger::instance().logCustom(
        t_params.logs_dir, t_params.acc_file,
        std::to_string(round) + " " + std::to_string(mngr.test_accuracy) + "\n");

    std::cout << "/// Server: Round " << round << " completed ///\n";
    Logger::instance().log("\n/// Server: Round " + std::to_string(round) + " completed ///\n");
  }

  Logger::instance().log("FINAL FLTRUST\n");
  mngr.updateModelParameters(w);
  mngr.runTesting();

  return w;
}