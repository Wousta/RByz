#include <cstdint>
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
#include "nets/cifar10Net.hpp"
#include "nets/mnistNet.hpp"
#include "nets/residual_block.hpp"
#include "nets/resnet.hpp"
#include "rbyzAux.hpp"
#include "rc_conn.hpp"
#include "rdmaOps.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"
#include "util.hpp"

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;
using namespace resnet;

/**
 * @brief Prepares RDMA registration configuration
 */
void prepareRdmaRegistration(int n_clients, 
                             TrainInputParams &t_params,
                             std::vector<RegInfo> &reg_info,
                             RegMemSrvr &regMem,
                             std::vector<ClientDataRbyz> &clnt_data_vec,
                             IRegDatasetMngr &mngr) {

  // Configure memory registration for each client
  for (int i = 0; i < n_clients; i++) {
    // Register memory addresses
    reg_info[i].addr_locs.push_back(castI(regMem.srvr_w));
    reg_info[i].addr_locs.push_back(castI(regMem.clnt_ws[i]));
    reg_info[i].addr_locs.push_back(castI(&regMem.srvr_ready_flag));
    reg_info[i].addr_locs.push_back(castI(&regMem.clnt_ready_flags[i]));
    reg_info[i].addr_locs.push_back(castI(&regMem.srvr_ready_rb));
    reg_info[i].addr_locs.push_back(castI(&clnt_data_vec[i].clnt_CAS));
    reg_info[i].addr_locs.push_back(castI(&clnt_data_vec[i].local_step));
    reg_info[i].addr_locs.push_back(castI(&clnt_data_vec[i].round));
    reg_info[i].addr_locs.push_back(castI(regMem.reg_data));

    if (!t_params.only_flt) {
      reg_info[i].addr_locs.push_back(castI(clnt_data_vec[i].forward_pass));
      reg_info[i].addr_locs.push_back(castI(clnt_data_vec[i].forward_pass_indices));
    }

    // Set memory sizes
    reg_info[i].data_sizes.push_back(regMem.reg_sz_data);
    reg_info[i].data_sizes.push_back(regMem.reg_sz_data);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(mngr.data_info.reg_data_size);

    if (!t_params.only_flt) {
      reg_info[i].data_sizes.push_back(clnt_data_vec[i].forward_pass_mem_size);
      reg_info[i].data_sizes.push_back(clnt_data_vec[i].forward_pass_indices_mem_size);
    }

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

  std::vector<int> result(allValues.begin(), allValues.begin() + size);
  std::sort(result.begin(), result.end());

  return result;
}

torch::Tensor aggregateUpdatesAvg(const std::vector<torch::Tensor> &client_updates,
                                  const torch::Tensor &server_update) {
  if (client_updates.empty()) {
    return server_update.clone(); // Return server update if no client updates
  }

  torch::Tensor aggregated_update = server_update.clone();
  for (const auto &update : client_updates) {
    aggregated_update += update;
  }
  aggregated_update /= static_cast<float>(client_updates.size() + 1); // +1 for server update
  //aggregated_update = torch::clamp(aggregated_update, -1.0f, 1.0f); // Clamp values to [-1, 1]
  return aggregated_update;
}

torch::Tensor aggregateUpdates(const std::vector<torch::Tensor> &client_updates,
                  const std::vector<torch::Tensor> &client_full_updates,
                  const torch::Tensor &server_update, 
                  const torch::Tensor &server_full_update,
                  const std::vector<uint32_t> &clnt_indices, 
                  std::vector<float> log_TS_vec, TrainInputParams &t_params) {

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
  float server_full_norm = torch::norm(server_full_update, 2).item<float>();
  for (int i = 0; i < client_updates.size(); i++) {
    // Compute cosine similarity with learnable parameters
    torch::Tensor dot_product = torch::dot(client_updates[i], server_update);
    float client_norm = torch::norm(client_updates[i], 2).item<float>();
    float cosine_sim = dot_product.item<float>() / (client_norm * server_norm);

    // Normalization of the full updates
    float client_full_norm = torch::norm(client_full_updates[i], 2).item<float>();

    // Apply ReLU (max with 0)
    float trust_score = std::max(0.0f, cosine_sim);
    trust_scores.push_back(trust_score);

    torch::Tensor normalized_update =
        client_full_updates[i] * (server_full_norm / client_full_norm);
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

  Logger::instance().logCustom(t_params.logs_dir, t_params.ts_file, "- Round end -\n");
  for (int i = 0; i < log_TS_vec.size(); i++) {
    std::string message = std::to_string(log_TS_vec[i]) + "\n";
    Logger::instance().logCustom(t_params.logs_dir, t_params.ts_file, message);
  }

  float sum_trust = 0.0f;
  for (float score : trust_scores) {
    sum_trust += score;
  }

  torch::Tensor aggregated_update = torch::zeros_like(server_full_update);
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

  Logger::instance().log("\nInitial weights gathered\n");
  std::vector<float> log_TS_vec(n_clients, 0.0f);

  for (int round = 1; round <= t_params.global_iters_fl; round++) {
    std::vector<int> polled_clients = generateRandomUniqueVector(n_clients, -1);
    tops::memcpyTensorVec(regMem.srvr_w, w, regMem.reg_sz_data);

    // Set the flag to indicate that the weights are ready for the clients to read
    regMem.srvr_ready_flag.store(round);

    // Run local training
    std::vector<torch::Tensor> w_pre_train = mngr.updateModelParameters(w);
    mngr.runTraining();
    std::vector<torch::Tensor> g = mngr.calculateUpdate(w_pre_train);
    
    // NOTE: RIGHT NOW EVERY CLIENT TRAINS AND READS THE AGGREGATED W IN EACH
    // ROUND, BUT SRVR ONLY READS FROM A RANDOM SUBSET OF CLIENTS
    std::vector<torch::Tensor> clnt_updates;
    std::vector<torch::Tensor> clnt_full_updates;
    std::vector<uint32_t> clnt_indices;
    clnt_updates.reserve(polled_clients.size());
    clnt_full_updates.reserve(polled_clients.size());
    clnt_indices.reserve(polled_clients.size());
    Logger::instance().log(
        "polled_clients size: " + std::to_string(polled_clients.size()) + ":\n");

    for (size_t i = 0; i < polled_clients.size(); i++) {
      int client = polled_clients[i];
      Logger::instance().log(
          "reading flags from client: " + std::to_string(client) + "\n");

      bool timed_out = false;
      long total_wait_time = 0;
      std::chrono::milliseconds initial_time(1);
      std::chrono::milliseconds limit_step_time(30000); // 30 seconds
      while (regMem.clnt_ready_flags[client] != round && !timed_out) {
        std::this_thread::sleep_for(initial_time);
        total_wait_time += initial_time.count();
        int64_t new_time = initial_time.count() * 3 / 2;
        initial_time = std::chrono::milliseconds(new_time);
        if (initial_time.count() % 5000 == 0) {
          Logger::instance().log("     Client " + 
                             std::to_string(client) + " ready flag: " + std::to_string(regMem.clnt_ready_flags[client]) + 
                             " waiting to be: " + std::to_string(round) + "\n");
        }
        if (initial_time > limit_step_time) {
          timed_out = true;
          Logger::instance().log("    -> Timeout in update gathering by client " + std::to_string(client) + 
                                 " for round " + std::to_string(round) + "\n");
        }
      }
      Logger::instance().log("    -> Server waited: " + std::to_string(total_wait_time) + " ms for client " +
                             std::to_string(client) + " ready flag: " + std::to_string(regMem.clnt_ready_flags[client]) + "\n");

      if (!timed_out) {
        size_t numel_server = regMem.reg_sz_data / sizeof(float);
        torch::Tensor flat_tensor = torch::from_blob(regMem.clnt_ws[client],
                            {static_cast<long>(numel_server)}, torch::kFloat32).clone();

        torch::Tensor learn_params = mngr.extractLearnableParams(flat_tensor);
        clnt_updates.push_back(learn_params);
        clnt_full_updates.push_back(flat_tensor);
        clnt_indices.push_back(client);
      }
    }

    // AGGREGATION PHASE //////////////////////
    torch::Tensor srvr_full_update = tops::flatten_tensor_vector(g);
    torch::Tensor srvr_update = mngr.extractLearnableParams(srvr_full_update);
    torch::Tensor aggregated_update =
        aggregateUpdates(clnt_updates, clnt_full_updates, 
                         srvr_update, srvr_full_update, clnt_indices, log_TS_vec, t_params); 
    std::vector<torch::Tensor> aggregated_update_vec =
        tops::reconstruct_tensor_vector(aggregated_update, w);

    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] + t_params.global_learn_rate * aggregated_update_vec[i];
    }
    mngr.updateModelParameters(w);
    Logger::instance().log("After aggregating: \n");
    mngr.runTesting();
    Logger::instance().logCustom(t_params.logs_dir, t_params.acc_file, std::to_string(round) + " " +
                                  std::to_string(mngr.test_accuracy) + "\n");

    std::cout << "///////////////// Server: Round " << round << " completed /////////////////\n";
    Logger::instance().log("\n//////////////// Server: Round " +
                           std::to_string(round) + " completed ////////////////\n");
  }

  Logger::instance().log("FINAL FLTRUST\n");
  mngr.updateModelParameters(w);
  mngr.runTesting();

  return w;
}

/**
 * @brief Allocates memory for the server and clients
 */
void allocateServerMemory(int n_clients, TrainInputParams &t_params, RegMemSrvr &regMem,
                          std::vector<ClientDataRbyz> &clnt_data_vec,
                          IRegDatasetMngr &mngr) {
  size_t sample_size = mngr.data_info.get_sample_size();
  std::vector<size_t> clnts_samples_count = mngr.getClientsSamplesCount(regMem.clnt_subset_size,
                                                                        regMem.srvr_subset_size,
                                                                        regMem.dataset_size);
                                                    
  for (int i = 0; i < n_clients; i++) {
    regMem.clnt_ws[i] = reinterpret_cast<float *>(malloc(regMem.reg_sz_data));
    clnt_data_vec[i].updates = regMem.clnt_ws[i];
    clnt_data_vec[i].index = i;
    size_t num_samples = clnts_samples_count[i];
    size_t batch_size = mngr.kTrainBatchSize;
    
    int64_t samples_fpass = num_samples;
    uint32_t total_batches = ceil(num_samples / batch_size);
    uint32_t batches_fpass = total_batches * t_params.batches_fpass_prop;
    if (t_params.batches_fpass_prop > 0) {
      samples_fpass = std::min(samples_fpass, batches_fpass * mngr.kTrainBatchSize);
      Logger::instance().log("Limiting forward pass samples to: " +
                            std::to_string(samples_fpass) + "\n");
    }

    // Calculate memory sizes for client data
    const size_t values_per_sample = mngr.f_pass_data.values_per_sample;
    const size_t bytes_per_value = mngr.f_pass_data.bytes_per_value;
    size_t forward_pass_mem_size =
        samples_fpass * values_per_sample * bytes_per_value;
    size_t forward_pass_indices_mem_size = samples_fpass * sizeof(uint32_t);

    // Set memory size information
    clnt_data_vec[i].num_samples = num_samples;
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
  Logger::instance().log("Pid: " + std::to_string(getpid()) + "\n");

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
      lyra::opt(t_params.logs_dir, "logs_dir")["--logs_dir"]("Directory for this experiment logs") |
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
      lyra::opt(t_params.label_flip_type, "label_flip_type")["--label_flip_type"]("Label flip type: 0 - no flip, 1 - random flip, 2-4 - targeted flips") |
      lyra::opt(t_params.flip_ratio, "flip_ratio")["--flip_ratio"]("Label flip ratio: 0.0 - 1.0") |
      lyra::opt(t_params.only_flt, "only_flt")["--only_flt"]("Run only FLTrust, no RByz") |
      lyra::opt(t_params.clnt_vd_proportion, "vd_prop")["--vd_prop"]("Proportion of validation data for each client") |
      lyra::opt(t_params.vd_prop_write, "vd_prop_write")["--vd_prop_write"]("Proportion of total chunks writable on client to write each time the test is renewed") |
      lyra::opt(t_params.test_renewal_freq, "test_renewal_freq")["--test_renewal_freq"]("Frequency of test renewal (every n rounds)") |
      lyra::opt(t_params.overwrite_poisoned, "overwrite_poisoned")["--overwrite_poisoned"]("Allow VD samples to overwrite poisoned samples") |
      lyra::opt(t_params.wait_all, "wait_all")["--wait_all"]("Ignore slow clients during RByz") |
      lyra::opt(t_params.batches_fpass_prop, "batches_fpass")["--batches_fpass"]("Number of batches for forward pass in RByz") |
      lyra::opt(t_params.srvr_wait_inc, "srvr_wait_inc")["--srvr_wait_inc"]("Increment for server wait time in timeouts experiment") |
      lyra::opt(t_params.extra_col_poison_prop, "extra_col_poison_prop")["--extra_col_poison_prop"]("Proportion of extra columns to poison in the dataset");

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
  std::cout << "Server address: srvr_ip = " << srvr_ip << ":" << port << "\n";
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
  std::cout << "VD proportion = " << t_params.clnt_vd_proportion << "\n";
  std::cout << "VD proportion write = " << t_params.vd_prop_write << "\n";
  std::cout << "Test renewal frequency = " << t_params.test_renewal_freq << "\n";
  std::cout << "Overwrite poisoned samples = " << (t_params.overwrite_poisoned ? "true" : "false") << "\n";
  std::cout << "Label flip type = " << t_params.label_flip_type << "\n";
  std::cout << "Label flip ratio = " << t_params.flip_ratio << "\n";
  std::cout << "Wait for all clients = " << (t_params.wait_all ? "true" : "false") << "\n";
  std::cout << "Batches for forward pass = " << t_params.batches_fpass_prop << "\n";
  std::cout << "Server wait increment = " << t_params.srvr_wait_inc << "\n";
  std::cout << "Proportion of extra columns to poison = " << t_params.extra_col_poison_prop << "\n";

  t_params.n_clients = n_clients; 
  MnistNet mnist_net;
  Cifar10Net cifar_net;
  std::array<int64_t, 3> layers{3, 3, 3};
  ResNet<ResidualBlock> resnet(layers, NUM_CLASSES);
  std::unique_ptr<RegMemSrvr> regMem;
  std::unique_ptr<IRegDatasetMngr> reg_mngr;

  if (t_params.use_mnist) {
    reg_mngr = std::make_unique<RegMnistMngr>(0, t_params, mnist_net);
    Logger::instance().log("Server: Using MNIST dataset\n");
  } else {
    reg_mngr = std::make_unique<RegCIFAR10Mngr>(0, t_params, cifar_net);
    // reg_mngr = std::make_unique<RegCIFAR10Mngr>(0, t_params, resnet);
    Logger::instance().log("Server: Using CIFAR10 dataset\n");
  }

  std::vector<torch::Tensor> dummy_w = reg_mngr->getInitialWeights();
  uint64_t reg_sz_data = 0;
  for (const auto& tensor : dummy_w) {
    reg_sz_data += tensor.numel() * sizeof(float);
  }

  regMem = std::make_unique<RegMemSrvr>(n_clients, reg_sz_data, reg_mngr->data_info.reg_data);
  regMem->dataset_size = reg_mngr->train_dataset_size;
  regMem->srvr_subset_size = t_params.srvr_subset_size;
  regMem->clnt_subset_size = t_params.clnt_subset_size;

  std::vector<RegInfo> reg_info(n_clients);
  std::vector<RcConn> conns(n_clients);
  std::vector<ClientDataRbyz> clnt_data_vec(n_clients);
  for (ClientDataRbyz &clnt_data : clnt_data_vec) {
    clnt_data.init(t_params.local_steps_rbyz);
  }
  allocateServerMemory(n_clients, t_params, *regMem, clnt_data_vec, *reg_mngr);
  prepareRdmaRegistration(n_clients, t_params, reg_info, *regMem, clnt_data_vec,
                          *reg_mngr);

  // Accept connection from each client
  for (int i = 0; i < n_clients; i++) {
    conns[i].acceptConn(addr_info, reg_info[i]);
    std::cout << "Connected to client " << i << "\n";
    Logger::instance().log("Connected to client " + std::to_string(i) + "\n");
  }

  std::string algo_name = (t_params.only_flt) ? "FLTrust" : "RByz";
  std::string dir = t_params.logs_dir;
  std::string final_data_file;
  int rounds;
  if (t_params.only_flt) {
    t_params.ts_file = "F_trust_scores.log";
    t_params.acc_file = "F_acc.log";
    t_params.included_agg_file = "F_included_agg.log";
    final_data_file = "F_final_data.log";
    rounds = t_params.global_iters_fl;
  } else {
    t_params.ts_file = "R_trust_scores.log";
    t_params.acc_file = "R_acc.log";
    t_params.included_agg_file = "R_included_agg.log";
    final_data_file = "R_final_data.log";
    rounds = t_params.global_iters_rbyz + t_params.global_iters_fl;
  }
  Logger::instance().logCustom(dir, t_params.ts_file, algo_name + "\n");
  Logger::instance().logCustom(dir, t_params.ts_file, std::to_string(rounds) + "\n");
  Logger::instance().logCustom(dir, t_params.ts_file, std::to_string(n_clients) + "\n");
  Logger::instance().logCustom(dir, t_params.ts_file, std::to_string(t_params.n_byz_clnts) + "\n");

  auto start = std::chrono::high_resolution_clock::now();
  RdmaOps rdma_ops(conns);
  RByzAux rbyz_aux(rdma_ops, *reg_mngr, t_params, clnt_data_vec);

  if (t_params.extra_col_poison_prop > 0.0f) {
    Logger::instance().log("Server: Extra column poisoning proportion: " + 
                           std::to_string(t_params.extra_col_poison_prop) + "\n");
    rbyz_aux.vdColAttack(t_params.extra_col_poison_prop);
  }

  std::vector<torch::Tensor> w = run_fltrust_srvr(
      n_clients, t_params, *reg_mngr, *regMem, clnt_data_vec);

  // Global rounds of RByz
  if (!t_params.only_flt) {
      rbyz_aux.runRByzServer(n_clients, w, *regMem);
  }

  auto end = std::chrono::high_resolution_clock::now();
  long elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  Logger::instance().log("Total time taken: " + std::to_string(elapsed) + " seconds\n");

  reg_mngr->runTesting();

  std::string acc_msg = "Accuracy " + std::to_string(reg_mngr->test_accuracy) + "\n";
  std::string time_msg = "Time " + std::to_string(elapsed) + "\n";
  std::string vd_msg = "clnt_vd_proportion " + std::to_string(t_params.clnt_vd_proportion) + "\n";
  std::string vd_prop_msg = "vd_prop_write " + std::to_string(t_params.vd_prop_write) + "\n";
  std::string test_renew_msg = "test_renewal_freq " + std::to_string(t_params.test_renewal_freq) + "\n";
  std::string recall_msg = "src_targ_class " + std::to_string(reg_mngr->src_class) + " " + 
                           std::to_string(reg_mngr->target_class) + "\n";
  std::string miss_samples_msg = "missclassed_samples " + std::to_string(reg_mngr->missclassed_samples) + "\n";
  std::string class_recall_msg = "src_class_recall " + std::to_string(reg_mngr->src_class_recall) + "\n";
  std::string rounds_to_converge_msg = "rounds_to_converge " + std::to_string(rbyz_aux.getRoundsToConverge()) + "\n";

  Logger::instance().logCustom(dir, final_data_file, acc_msg);
  Logger::instance().logCustom(dir, final_data_file, time_msg);
  Logger::instance().logCustom(dir, final_data_file, recall_msg);
  Logger::instance().logCustom(dir, final_data_file, miss_samples_msg);
  Logger::instance().logCustom(dir, final_data_file, class_recall_msg);
  Logger::instance().logCustom(dir, final_data_file, rounds_to_converge_msg);
  
  if (!t_params.only_flt) {
    Logger::instance().logCustom(dir, final_data_file, vd_msg);
    Logger::instance().logCustom(dir, final_data_file, vd_prop_msg);
    Logger::instance().logCustom(dir, final_data_file, test_renew_msg);
  }

  Logger::instance().logCustom(dir, final_data_file, "$ END OF EXECUTION $\n");
  Logger::instance().logCustom(dir, t_params.acc_file, "$ END OF EXECUTION $\n");
  Logger::instance().logCustom(dir, t_params.ts_file, "$ END OF EXECUTION $\n");
  Logger::instance().logCustom(dir, t_params.included_agg_file, "$ END OF EXECUTION $\n");
  Logger::instance().logCustom(dir, t_params.clnts_renew_file, "$ END OF EXECUTION $\n");

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
