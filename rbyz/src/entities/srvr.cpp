#include <float.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iostream>
#include <lyra/lyra.hpp>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "rc_conn.hpp"
#include "rdma-api.hpp"
#include "util.hpp"
#include "attacks.hpp"
#include "global/globalConstants.hpp"
#include "global/logger.hpp"
#include "datasetLogic/registeredMnistTrain.hpp"
#include "datasetLogic/regularMnistTrain.hpp"
#include "datasetLogic/baseMnistTrain.hpp"
#include "rbyzAux.hpp"
#include "rdmaOps.hpp"
#include "tensorOps.hpp"

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

/**
 * @brief Prepares RDMA registration configuration
 */
void prepareRdmaRegistration(
    int n_clients,
    std::vector<RegInfo>& reg_info,
    RegMemSrvr &regMem,
    std::vector<ClientDataRbyz>& clnt_data_vec,
    RegisteredMnistTrain &registered_mnist) {

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
    reg_info[i].addr_locs.push_back(castI(registered_mnist.getRegisteredData()));
    reg_info[i].addr_locs.push_back(castI(clnt_data_vec[i].forward_pass));
    reg_info[i].addr_locs.push_back(castI(clnt_data_vec[i].forward_pass_indices));

    // Set memory sizes
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(REG_SZ_DATA);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(REG_SZ_DATA);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(registered_mnist.getRegisteredDataSize());
    reg_info[i].data_sizes.push_back(clnt_data_vec[i].forward_pass_mem_size);
    reg_info[i].data_sizes.push_back(clnt_data_vec[i].forward_pass_indices_mem_size);

    // Set permissions for remote access
    reg_info[i].permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                            IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  }
}

std::vector<int> generateRandomUniqueVector(int n_clients, int min_sz = -1) {
  if (min_sz == -1) {
    min_sz = n_clients;
  }

  std::cout << "Generating random unique vector of size " << n_clients << "\n";

  // Initialize random number generator
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

torch::Tensor aggregate_updates(const std::vector<torch::Tensor> &client_updates,
                                const torch::Tensor &server_update) {
  // Compute cosine similarity between each client update and server update
  std::vector<float> trust_scores;
  std::vector<torch::Tensor> normalized_updates;
  trust_scores.reserve(client_updates.size());
  normalized_updates.reserve(client_updates.size());

  for (const auto &flat_client_update : client_updates) {
    // Compute cosine similarity
    torch::Tensor dot_product = torch::dot(flat_client_update, server_update);
    float client_norm = torch::norm(flat_client_update, 2).item<float>();
    float server_norm = torch::norm(server_update, 2).item<float>();
    float cosine_sim = dot_product.item<float>() / (client_norm * server_norm);

    // Apply ReLU (max with 0)
    float trust_score = std::max(0.0f, cosine_sim);
    trust_scores.push_back(trust_score);

    torch::Tensor normalized_update = flat_client_update * (server_norm / client_norm);
    normalized_updates.push_back(normalized_update);
  }

  {
    std::ostringstream oss;
    oss << "\nTRUST SCORES:\n";
    for (int i = 0; i < trust_scores.size(); i++) {
      if (i % 1 == 0) {
        oss << "  Client " << i << " score: " << trust_scores[i] << "\n";
      }
    }
    Logger::instance().log(oss.str());
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

  // Print the aggregated update
  {
    std::ostringstream oss;
    oss << "\nAggregated update:\n";
    oss << "  "
        << aggregated_update.slice(0, 0, std::min<size_t>(aggregated_update.numel(), 5)).toString()
        << "\n";
    Logger::instance().log(oss.str());
  }

  // If all trust scores are 0, just return the zero tensor
  return aggregated_update;
}

std::vector<torch::Tensor> run_fltrust_srvr(int rounds,
                                            int n_clients,
                                            BaseMnistTrain &mnist,
                                            RegMemSrvr &regMem,
                                            std::vector<ClientDataRbyz> &clntsData) {
  std::vector<torch::Tensor> w = mnist.getInitialWeights();
  printTensorSlices(w, 0, 5);
  Logger::instance().log("\nInitial run of minstrain done\n");

  for (int round = 1; round <= rounds; round++) {
    auto all_tensors = flatten_tensor_vector(w);
    std::vector<int> polled_clients = generateRandomUniqueVector(n_clients);
    std::vector<torch::Tensor> clnt_updates;

    // Copy to shared memory
    size_t total_bytes = all_tensors.numel() * sizeof(float);
    float *global_w = all_tensors.data_ptr<float>();
    std::memcpy(regMem.srvr_w, global_w, total_bytes);

    // Set the flag to indicate that the weights are ready for the clients to read
    regMem.srvr_ready_flag = round;

    // Run local training
    std::vector<torch::Tensor> g = mnist.runMnistTrain(round, w);

    // Keep updated values to follow FLtrust logic
    for (size_t i = 0; i < g.size(); ++i) {
      g[i] -= w[i];
    }

    Logger::instance().log("Weight updates:\n");
    printTensorSlices(g, 0, 5);

    // NOTE: RIGHT NOW EVERY CLIENT TRAINS AND READS THE AGGREGATED W IN EACH ROUND,
    // BUT SRVR ONLY READS FROM A RANDOM SUBSET OF CLIENTS
    Logger::instance().log("polled_clients size: " + std::to_string(polled_clients.size()) + "\n");
    {
      std::ostringstream oss;
      oss << "Clients polled: \n";
      for (int index : polled_clients) {
        oss << index << " ";
      }
      oss << "\n";
      Logger::instance().log(oss.str());
    }
    for (size_t i = 0; i < polled_clients.size(); i++) {
      int client = polled_clients[i];
      Logger::instance().log("reading flags from client: " + std::to_string(client) + "\n");
      while (regMem.clnt_ready_flags[client] != round) {
        std::this_thread::yield();
      }

      size_t numel_server = REG_SZ_DATA / sizeof(float);
      torch::Tensor flat_tensor =
          torch::from_blob(
              regMem.clnt_ws[client], {static_cast<long>(numel_server)}, torch::kFloat32)
              .clone();

      clnt_updates.push_back(flat_tensor);
    }

    // Use attacks to simulate Byzantine clients
    clnt_updates =
        no_byz(clnt_updates, mnist.getModel(), GLOBAL_LEARN_RATE, N_BYZ_CLNTS, mnist.getDevice());
    // clnt_updates = trim_attack(
    //   clnt_updates,
    //   mnist.getModel(),
    //   GLOBAL_LEARN_RATE,
    //   N_BYZ_CLNTS,
    //   mnist.getDevice()
    // );

    Logger::instance().log("Server: Done with Byzantine attack\n");

    // AGGREGATION PHASE //////////////////////
    torch::Tensor flat_srvr_update = flatten_tensor_vector(g);
    torch::Tensor aggregated_update = aggregate_updates(clnt_updates, flat_srvr_update);
    std::vector<torch::Tensor> aggregated_update_vec =
        reconstruct_tensor_vector(aggregated_update, w);
    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] + GLOBAL_LEARN_RATE * aggregated_update_vec[i];
    }
  }

  return w;
}

/**
 * @brief Allocates memory for the server and clients
 */
void allocateServerMemory(
    int n_clients,
    RegMemSrvr &regMem,
    std::vector<ClientDataRbyz>& clnt_data_vec,
    RegisteredMnistTrain &registered_mnist) {

  size_t sample_size = registered_mnist.getSampleSize();
  std::vector<size_t> clnts_samples_count = registered_mnist.getClientsSamplesCount();

  for (int i = 0; i < n_clients; i++) {
    // Allocate memory for client weights and metrics
    regMem.clnt_ws[i] = reinterpret_cast<float *>(malloc(REG_SZ_DATA));
    regMem.clnt_loss_and_err[i] = reinterpret_cast<float *>(malloc(MIN_SZ));

    // Set up client data structure
    clnt_data_vec[i].clnt_index = i;

    // Shared memory locations with FLtrust
    clnt_data_vec[i].updates = regMem.clnt_ws[i];
    clnt_data_vec[i].loss = regMem.clnt_loss_and_err[i];
    clnt_data_vec[i].error_rate = regMem.clnt_loss_and_err[i] + 1;

    // Calculate memory sizes for client data
    size_t num_samples = clnts_samples_count[i];
    const size_t values_per_sample = registered_mnist.getValuesPerSample();
    const size_t bytes_per_value = registered_mnist.getBytesPerValue();
    size_t forward_pass_mem_size = num_samples * values_per_sample * bytes_per_value;
    size_t forward_pass_indices_mem_size = num_samples * sizeof(uint32_t);
    
    // Set memory size information
    clnt_data_vec[i].dataset_size = num_samples * sample_size;
    clnt_data_vec[i].forward_pass_mem_size = forward_pass_mem_size;
    clnt_data_vec[i].forward_pass_indices_mem_size = forward_pass_indices_mem_size;

    // Allocate memory for forward pass data
    clnt_data_vec[i].forward_pass = reinterpret_cast<float *>(malloc(forward_pass_mem_size));
    clnt_data_vec[i].forward_pass_indices = reinterpret_cast<uint32_t *>(malloc(forward_pass_indices_mem_size));
  }
}

int main(int argc, char *argv[]) {
  Logger::instance().log("Server starting RBYZ\n");
  int n_clients;
  bool load_model = false;
  std::string model_file = "mnist_model_params.pt";
  std::string srvr_ip;
  std::string port;
  unsigned int posted_wqes;
  AddrInfo addr_info;
  std::shared_ptr<ltncyVec> latency = std::make_shared<ltncyVec>();
  latency->reserve(10);
  auto cli = lyra::cli() | lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
             lyra::opt(port, "port")["-p"]["--port"]("port") |
             lyra::opt(n_clients, "n_clients")["-w"]["--n_clients"]("n_clients") |
             lyra::opt(load_model)["-l"]["--load"]("Load model from saved file") |
             lyra::opt(model_file, "model_file")["-f"]["--file"]("Model file path");

  auto result = cli.parse({argc, argv});
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
    return 1;
  }

  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());
  std::cout << "Server: n_clients = " << n_clients << "\n";
  std::cout << "Server: srvr_ip = " << srvr_ip << "\n";
  std::cout << "Server: port = " << port << "\n";

  // Objects for training fltrust and rbyz
  std::unique_ptr<BaseMnistTrain> regular_mnist = 
    std::make_unique<RegularMnistTrain>(0, n_clients + 1, SRVR_SUBSET_SIZE);
  std::unique_ptr<RegisteredMnistTrain> registered_mnist =
    std::make_unique<RegisteredMnistTrain>(0, n_clients + 1, SRVR_SUBSET_SIZE);

  // Data structures for connection and data registration
  std::vector<RegInfo> reg_info(n_clients);
  std::vector<RcConn> conns(n_clients);
  std::vector<comm_info> conn_data;

  // Data structures for server and clients registered memory
  RegMemSrvr regMem(n_clients);
  std::vector<ClientDataRbyz> clnt_data_vec(n_clients);
  allocateServerMemory(n_clients, regMem, clnt_data_vec, *registered_mnist);
  prepareRdmaRegistration(n_clients, reg_info, regMem, clnt_data_vec, *registered_mnist);
  
  // Accept connection from each client
  for (int i = 0; i < n_clients; i++) {
    conns[i].acceptConn(addr_info, reg_info[i]);
    conn_data.push_back(conns[i].getConnData());
    std::cout << "Connected to client " << i << "\n";
  }

  std::vector<torch::Tensor> w;
  if (load_model) {
    w = regular_mnist->loadModelState(model_file);
    if (w.empty()) {
      Logger::instance().log("Failed to load model state. Running FLTrust instead.\n");
      std::cout << "Failed to load model state from file: " << model_file << "\n";
      load_model = false;
    } else {
      Logger::instance().log("Successfully loaded model from file.\n");
      std::cout << "Loaded model state from file: " << model_file << "\n";

      // Do one iteration of fltrust with ALL clients to initialize trust scores
      std::cout << "SRVR Running FLTrust with loaded model\n";
      w = run_fltrust_srvr(1, n_clients, *regular_mnist, regMem, clnt_data_vec);
      Logger::instance().log("SRVR FLTrust with loaded model done\n");
    }
  }

  if (!load_model) {
    Logger::instance().log("SRVR Running FLTrust, load model set FALSE\n");
    w = run_fltrust_srvr(GLOBAL_ITERS, n_clients, *regular_mnist, regMem, clnt_data_vec);

    // regular_mnist.saveModelState(w, model_file);
  }

  // Global rounds of RByz
  RdmaOps rdma_ops(conn_data);
  registered_mnist->copyModelParameters(regular_mnist->getModel());
  runRByzServer(n_clients, w, *registered_mnist, rdma_ops, regMem, clnt_data_vec);

  for (RcConn conn : conns) {
    conn.disconnect();
  }
  free(addr_info.ipv4_addr);
  free(addr_info.port);

  std::cout << "\nServer done\n";

  // sleep for server to be available
  Logger::instance().log("\nSleeping for 1 hour\n");

  std::this_thread::sleep_for(std::chrono::hours(1));
  return 0;
}
