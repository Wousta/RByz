#include <float.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iostream>
#include <lyra/lyra.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "attacks.hpp"
#include "datasetLogic/iRegDatasetMngr.hpp"
#include "datasetLogic/regCIFAR10Mngr.hpp"
#include "datasetLogic/regMnistMngr.hpp"
#include "entities.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "nets/cifar10Net.hpp"
#include "nets/mnistNet.hpp"
#include "nets/residual_block.hpp"
#include "nets/resnet.hpp"
#include "rbyzAux.hpp"
#include "rc_conn.hpp"
#include "rdmaOps.hpp"
#include "tensorOps.hpp"
#include "util.hpp"

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;
using namespace resnet;

/**
 * @brief Prepares RDMA registration configuration
 */
void prepareRdmaRegistration(int n_clients, std::vector<RegInfo> &reg_info,
                             RegMemSrvr &regMem,
                             std::vector<ClientDataRbyz> &clnt_data_vec,
                             IRegDatasetMngr &mngr) {

  // Configure memory registration for each client
  for (int i = 0; i < n_clients; i++) {
    // Register memory addresses
    reg_info[i].addr_locs.push_back(castI(&regMem.srvr_ready_flag));
    reg_info[i].addr_locs.push_back(castI(regMem.srvr_w));
    reg_info[i].addr_locs.push_back(castI(&regMem.clnt_ready_flags[i]));
    reg_info[i].addr_locs.push_back(castI(regMem.clnt_ws[i]));
    reg_info[i].addr_locs.push_back(castI(regMem.clnt_loss_and_err[i]));
    reg_info[i].addr_locs.push_back(castI(&clnt_data_vec[i].clnt_CAS));
    reg_info[i].addr_locs.push_back(castI(&clnt_data_vec[i].local_step));
    reg_info[i].addr_locs.push_back(castI(&clnt_data_vec[i].round));
    reg_info[i].addr_locs.push_back(castI(regMem.reg_data));
    reg_info[i].addr_locs.push_back(castI(clnt_data_vec[i].forward_pass));
    reg_info[i].addr_locs.push_back(castI(clnt_data_vec[i].forward_pass_indices));

    // Set memory sizes
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(regMem.reg_sz_data);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(regMem.reg_sz_data);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(mngr.data_info.reg_data_size);
    reg_info[i].data_sizes.push_back(clnt_data_vec[i].forward_pass_mem_size);
    reg_info[i].data_sizes.push_back(clnt_data_vec[i].forward_pass_indices_mem_size);

    // Set permissions for remote access
    reg_info[i].permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                              IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_ATOMIC;
  }
}

std::vector<int> generateRandomUniqueVector(int n_clients, int min_sz) {
  if (min_sz == -1) {
    min_sz = n_clients;
  }
  std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
  std::vector<int> allValues(n_clients);
  for (int i = 0; i < n_clients; i++) {
    allValues[i] = i;
  }
  std::shuffle(allValues.begin(), allValues.end(), rng);

  // Generate random size
  std::uniform_int_distribution<int> sizeDist(min_sz, n_clients);
  int size = sizeDist(rng);

  std::cout << "Random size: " << size << "\n";

  std::vector<int> result(allValues.begin(), allValues.begin() + size);
  std::sort(result.begin(), result.end());

  return result;
}

torch::Tensor
aggregate_updates(const std::vector<torch::Tensor> &client_updates,
                  const torch::Tensor &server_update, std::vector<uint32_t> &clnt_indices, std::vector<float> log_TS_vec, int only_flt) {

  if (client_updates.empty()) {
    Logger::instance().log(
        "No client updates provided, returning server update.\n");
    return server_update.clone(); // Return server update if no client updates
  }

  // Compute cosine similarity between each client update and server update
  std::vector<float> trust_scores;
  std::vector<torch::Tensor> normalized_updates;
  trust_scores.reserve(client_updates.size());
  normalized_updates.reserve(client_updates.size());

  float server_norm = torch::norm(server_update, 2).item<float>();
  for (const auto &flat_client_update : client_updates) {
    // Compute cosine similarity
    torch::Tensor dot_product = torch::dot(flat_client_update, server_update);
    float client_norm = torch::norm(flat_client_update, 2).item<float>();
    float cosine_sim = dot_product.item<float>() / (client_norm * server_norm);

    // Apply ReLU (max with 0)
    float trust_score = std::max(0.0f, cosine_sim);
    trust_scores.push_back(trust_score);

    torch::Tensor normalized_update =
        flat_client_update * (server_norm / client_norm);
    normalized_updates.push_back(normalized_update);
  }

  {
    std::ostringstream oss;
    oss << "\nTRUST SCORES FLtrust:\n";
    for (int i = 0; i < trust_scores.size(); i++) {
      log_TS_vec[clnt_indices[i]] = trust_scores[i];
      oss << "  Client " << i << " score: " << trust_scores[i] << "\n";
    }
    Logger::instance().log(oss.str());
  }

  std::string filename = (only_flt) ? "F_trust_scores.log" : "R_trust_scores.log";
  Logger::instance().logCustom("", filename, "- FL\n");
  for (int i = 0; i < log_TS_vec.size(); i++) {
    std::string message = std::to_string(log_TS_vec[i]) + "\n";
    Logger::instance().logCustom("", filename, message);
  }

  // Normalize trust scores
  float sum_trust = 0.0f;
  for (float score : trust_scores) {
    sum_trust += score;
  }

  torch::Tensor aggregated_update = torch::zeros_like(server_update);
  if (sum_trust > 0) {
    for (int i = 0; i < trust_scores.size(); i++) {
      aggregated_update += trust_scores[i] * normalized_updates[i];
    }

    aggregated_update /= sum_trust;
  }

  // If all trust scores are 0, just return the zero tensor
  return aggregated_update;
}

std::vector<torch::Tensor>
run_fltrust_srvr(int n_clients, TrainInputParams t_params, IRegDatasetMngr &mngr,
                 RegMemSrvr &regMem, std::vector<ClientDataRbyz> &clntsData) {
  std::vector<torch::Tensor> w = mngr.getInitialWeights();
  printTensorSlices(w, 0, 5);
  Logger::instance().log("\nInitial run of minstrain done\n");
  std::vector<float> log_TS_vec(n_clients, 0.0f);

  for (int round = 1; round <= t_params.global_iters_fl; round++) {
    auto all_tensors = flatten_tensor_vector(w);
    std::vector<int> polled_clients = generateRandomUniqueVector(n_clients, -1);

    // Copy to shared memory
    size_t total_bytes = all_tensors.numel() * sizeof(float);
    float *global_w = all_tensors.data_ptr<float>();
    std::memcpy(regMem.srvr_w, global_w, total_bytes);

    // Set the flag to indicate that the weights are ready for the clients to
    // read
    regMem.srvr_ready_flag = round;

    // Run local training
    Logger::instance().log("Server: Running FLtrust training for round " +
                           std::to_string(round) + "\n");
    std::vector<torch::Tensor> g = mngr.runTraining(round, w);

    // NOTE: RIGHT NOW EVERY CLIENT TRAINS AND READS THE AGGREGATED W IN EACH
    // ROUND, BUT SRVR ONLY READS FROM A RANDOM SUBSET OF CLIENTS
    std::vector<torch::Tensor> clnt_updates;
    std::vector<uint32_t> clnt_indices;
    clnt_updates.reserve(polled_clients.size());
    Logger::instance().log(
        "polled_clients size: " + std::to_string(polled_clients.size()) + "\n");

    for (size_t i = 0; i < polled_clients.size(); i++) {
      int client = polled_clients[i];
      Logger::instance().log(
          "reading flags from client: " + std::to_string(client) + "\n");

      bool timed_out = false;
      long total_wait_time = 0;
      std::chrono::microseconds initial_time(20); // time of 10 round trips
      std::chrono::microseconds limit_step_time(200000000); // 200 milliseconds
      while (regMem.clnt_ready_flags[client] != round && !timed_out) {
        std::this_thread::sleep_for(initial_time);
        total_wait_time += initial_time.count();
        initial_time *= 2; // Exponential backoff
        if (initial_time > limit_step_time) {
          timed_out = true;
          Logger::instance().log("    -> Timeout in update gathering by client " + std::to_string(client) + 
                                 " for round " + std::to_string(round) + "\n");
        }
      }
      Logger::instance().log("    -> Server waited: " + std::to_string(total_wait_time) + " us for client " + 
                             std::to_string(client) + "\n");

      if (!timed_out) {
        size_t numel_server = regMem.reg_sz_data / sizeof(float);
        torch::Tensor flat_tensor =
            torch::from_blob(regMem.clnt_ws[client],
                            {static_cast<long>(numel_server)}, torch::kFloat32).clone();

        clnt_updates.push_back(flat_tensor);
        clnt_indices.push_back(client);
      }
    }

    // AGGREGATION PHASE //////////////////////
    torch::Tensor flat_srvr_update = flatten_tensor_vector(g);
    torch::Tensor aggregated_update =
        aggregate_updates(clnt_updates, flat_srvr_update, clnt_indices, log_TS_vec, t_params.only_flt);
    std::vector<torch::Tensor> aggregated_update_vec =
        reconstruct_tensor_vector(aggregated_update, w);

    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] + t_params.global_learn_rate * aggregated_update_vec[i];
    }
    mngr.updateModelParameters(w);
    Logger::instance().log("After aggregating: \n");
    mngr.runTesting();
    Logger::instance().logAcc(t_params.only_flt, std::to_string(round) + " " +
                                  std::to_string(mngr.test_accuracy) + "\n");
  }

  Logger::instance().log("FINAL FLTRUST\n");
  mngr.updateModelParameters(w);
  mngr.runTesting();

  return w;
}

/**
 * @brief Allocates memory for the server and clients
 */
void allocateServerMemory(int n_clients, RegMemSrvr &regMem,
                          std::vector<ClientDataRbyz> &clnt_data_vec,
                          IRegDatasetMngr &mngr) {

  size_t sample_size = mngr.data_info.get_sample_size();
  std::vector<size_t> clnts_samples_count = mngr.getClientsSamplesCount(regMem.clnt_subset_size,
                                                                        regMem.srvr_subset_size,
                                                                        regMem.dataset_size);

  for (int i = 0; i < n_clients; i++) {
    // Allocate memory for client weights and metrics
    regMem.clnt_ws[i] = reinterpret_cast<float *>(malloc(regMem.reg_sz_data));
    regMem.clnt_loss_and_err[i] = reinterpret_cast<float *>(malloc(MIN_SZ));

    // Set up client data structure
    clnt_data_vec[i].index = i;

    // Shared memory locations with FLtrust
    clnt_data_vec[i].updates = regMem.clnt_ws[i];
    clnt_data_vec[i].loss = regMem.clnt_loss_and_err[i];
    clnt_data_vec[i].error_rate = regMem.clnt_loss_and_err[i] + 1;

    // Calculate memory sizes for client data
    size_t num_samples = clnts_samples_count[i];
    size_t batch_size = mngr.kTrainBatchSize;
    const size_t values_per_sample = mngr.f_pass_data.values_per_sample;
    const size_t bytes_per_value = mngr.f_pass_data.bytes_per_value;
    size_t forward_pass_mem_size =
        num_samples * values_per_sample * bytes_per_value;
    size_t forward_pass_indices_mem_size = num_samples * sizeof(uint32_t);

    // Set memory size information
    clnt_data_vec[i].dataset_size = num_samples * sample_size;
    clnt_data_vec[i].forward_pass_mem_size = forward_pass_mem_size;
    clnt_data_vec[i].forward_pass_indices_mem_size =
        forward_pass_indices_mem_size;

    // Allocate memory for forward pass data
    clnt_data_vec[i].forward_pass =
        reinterpret_cast<float *>(malloc(forward_pass_mem_size));
    clnt_data_vec[i].forward_pass_indices =
        reinterpret_cast<uint32_t *>(malloc(forward_pass_indices_mem_size));
  }
}

int main(int argc, char *argv[]) {
  Logger::instance().log(
      "Server starting RBYZ in core: " + std::to_string(sched_getcpu()) + "\n");

  TrainInputParams t_params;
  int n_clients;
  std::string srvr_ip;
  std::string port;
  unsigned int posted_wqes;
  AddrInfo addr_info;
  std::shared_ptr<ltncyVec> latency = std::make_shared<ltncyVec>();
  latency->reserve(10);
  auto cli =
      lyra::cli() |
      lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
      lyra::opt(port, "port")["-p"]["--port"]("port") |
      lyra::opt(n_clients, "n_clients")["-w"]["--n_clients"]("n_clients") |
      lyra::opt(t_params.use_mnist)["-l"]["--load"]("Load model from saved file") |
      lyra::opt(t_params.n_byz_clnts, "n_byz")["-b"]["--n_byz"]("byzantine clients") |
      lyra::opt(t_params.epochs, "epochs")["--epochs"]("number of epochs") |
      lyra::opt(t_params.batch_size, "batch_size")["--batch_size"]("batch size") |
      lyra::opt(t_params.global_learn_rate, "global_learn_rate")["--global_learn_rate"]("global learning rate") |
      lyra::opt(t_params.local_learn_rate, "local_learn_rate")["--local_learn_rate"]("global learning rate") |
      lyra::opt(t_params.clnt_subset_size, "clnt_subset_size")["--clnt_subset_size"]("client subset size") |
      lyra::opt(t_params.srvr_subset_size, "srvr_subset_size")["--srvr_subset_size"]("server subset size") |
      lyra::opt(t_params.global_iters_fl, "global_iters_fl")["--global_iters_fl"]("global iterations FL") |
      lyra::opt(t_params.local_steps_rbyz, "local_steps_rbyz")["--local_steps_rbyz"]("local steps RByz") |
      lyra::opt(t_params.global_iters_rbyz, "global_iters_rbyz")["--global_iters_rbyz"]("global iterations RByz") |
      lyra::opt(t_params.chunk_size, "chunk_size")["--chunk_size"]("chunk size for VDsampling") |
      lyra::opt(t_params.only_flt, "only_flt")["--only_flt"]("Run only FLTrust, no RByz") |
      lyra::opt(t_params.vd_proportion, "vd_prop")["--vd_prop"]("Proportion of validation data for each client");

  auto result = cli.parse({argc, argv});
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
              << std::endl;
    return 1;
  }

  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());
  std::cout << "Dataset type: " << (t_params.use_mnist ? "MNIST" : "CIFAR10") << "\n";
  std::cout << "Server: n_clients = " << n_clients << "\n";
  std::cout << "Server: srvr_ip = " << srvr_ip << "\n";
  std::cout << "Server: port = " << port << "\n";
  std::cout << "Byz clients = " << t_params.n_byz_clnts << "\n";
  std::cout << "Batch size = " << t_params.batch_size << "\n";
  std::cout << "Global learn rate = " << t_params.global_learn_rate << "\n";
  std::cout << "Local learn rate = " << t_params.local_learn_rate << "\n";
  std::cout << "Client subset size = " << t_params.clnt_subset_size << "\n";
  std::cout << "Server subset size = " << t_params.srvr_subset_size << "\n";
  std::cout << "Global iterations FL = " << t_params.global_iters_fl << "\n";
  std::cout << "Local steps RByz = " << t_params.local_steps_rbyz << "\n";
  std::cout << "Global iterations RByz = " << t_params.global_iters_rbyz << "\n";
  std::cout << "Only FLTrust = " << (t_params.only_flt ? "true" : "false") << "\n";
  std::cout << "VD proportion = " << t_params.vd_proportion << "\n";

  t_params.num_workers = n_clients + 1; // +1 for server
  MnistNet mnist_net;
  Cifar10Net cifar_net;
  std::array<int64_t, 3> layers{2, 2, 2};
  ResNet<ResidualBlock> resnet(layers, NUM_CLASSES);
  std::unique_ptr<RegMemSrvr> regMem;
  std::unique_ptr<IRegDatasetMngr> reg_mngr;

  if (t_params.use_mnist) {
    reg_mngr = std::make_unique<RegMnistMngr>(0, t_params, mnist_net);

    regMem = std::make_unique<RegMemSrvr>(n_clients, REG_SZ_DATA_MNIST, reg_mngr->data_info.reg_data);
    regMem->dataset_size = DATASET_SIZE_MNIST;
    Logger::instance().log("Server: Using MNIST dataset\n");
  } else {
    reg_mngr = std::make_unique<RegCIFAR10Mngr>(0, t_params, resnet);

    std::vector<torch::Tensor> dummy = reg_mngr->getInitialWeights();
    uint64_t reg_sz_data = 0;
    for (const auto& tensor : dummy) {
      reg_sz_data += tensor.numel() * sizeof(float);
    }

    regMem = std::make_unique<RegMemSrvr>(n_clients, reg_sz_data, reg_mngr->data_info.reg_data);
    regMem->dataset_size = DATASET_SIZE_CF10;
    Logger::instance().log("Server: Using CIFAR10 dataset\n");
  }
  regMem->srvr_subset_size = t_params.srvr_subset_size;
  regMem->clnt_subset_size = t_params.clnt_subset_size;

  std::vector<RegInfo> reg_info(n_clients);
  std::vector<RcConn> conns(n_clients);
  std::vector<ClientDataRbyz> clnt_data_vec(n_clients);
  for (ClientDataRbyz &clnt_data : clnt_data_vec) {
    clnt_data.init(t_params.local_steps_rbyz);
  }
  allocateServerMemory(n_clients, *regMem, clnt_data_vec, *reg_mngr);
  prepareRdmaRegistration(n_clients, reg_info, *regMem, clnt_data_vec,
                          *reg_mngr);

  // Accept connection from each client
  for (int i = 0; i < n_clients; i++) {
    conns[i].acceptConn(addr_info, reg_info[i]);
    std::cout << "Connected to client " << i << "\n";
  }

  if (t_params.only_flt) {
    Logger::instance().log("Server: Running FLTrust only\n");
    Logger::instance().openFLAccLog();
  } else {
    Logger::instance().log("Server: Running RByz\n");
    Logger::instance().openRByzAccLog();
  }
  auto start = std::chrono::high_resolution_clock::now();

  std::string filename;
  int rounds;
  if (t_params.only_flt) {
    filename = "F_trust_scores.log";
    rounds = t_params.global_iters_fl;
  } else {
    filename = "R_trust_scores.log";
    rounds = t_params.global_iters_rbyz + t_params.global_iters_fl;
  }
  Logger::instance().logCustom("", filename, std::to_string(rounds) + "\n");
  Logger::instance().logCustom("", filename, std::to_string(n_clients) + "\n");

  std::cout << "SRVR Running FLTrust\n";
  std::vector<torch::Tensor> w = run_fltrust_srvr(
      n_clients, t_params, *reg_mngr, *regMem, clnt_data_vec);
  // auto end = std::chrono::high_resolution_clock::now();
  // Logger::instance().log("Total time taken: " +
  //                        std::to_string(std::chrono::duration_cast<std::chrono::seconds>(end
  //                        - start).count()) + " seconds\n");

  // Global rounds of RByz
  RdmaOps rdma_ops(conns);
  Logger::instance().log("Initial test of the model before RByz\n");

  RByzAux rbyz_aux(rdma_ops, *reg_mngr, t_params);
  if (!t_params.only_flt) {
      rbyz_aux.runRByzServer(n_clients, w, *regMem, clnt_data_vec);
  }

  auto end = std::chrono::high_resolution_clock::now();
  long elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  Logger::instance().log("Total time taken: " + std::to_string(elapsed) + " seconds\n");

  Logger::instance().logCustom("", filename, "$ END OF EXECUTION $\n");
  reg_mngr->runTesting();
  std::string final_data_file = (t_params.only_flt) ? "F_final_data.log" : "R_final_data.log";
  Logger::instance().logCustom("", final_data_file, std::to_string(t_params.vd_proportion) + "\n");
  Logger::instance().logCustom("", final_data_file, std::to_string(reg_mngr->test_accuracy) + "\n");
  Logger::instance().logCustom("", final_data_file, std::to_string(elapsed) + "\n");
  Logger::instance().logCustom("", final_data_file, "$ END OF EXECUTION $\n");

  rbyz_aux.awaitTermination(clnt_data_vec, t_params.global_iters_rbyz);
  regMem->srvr_ready_flag = SRVR_FINISHED;
  for (int i = 0; i < n_clients; i++) {
    rdma_ops.exec_rdma_write( sizeof(int), SRVR_READY_IDX, i);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    conns[i].disconnect();
  }

  free(addr_info.ipv4_addr);
  free(addr_info.port);

  std::cout << "\nServer done\n";
  Logger::instance().log("\nServer done\n");

  std::this_thread::sleep_for(std::chrono::seconds(1));
  return 0;
}
