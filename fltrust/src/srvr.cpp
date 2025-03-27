#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "../include/mnistTrain.hpp"
#include "../include/globalConstants.hpp"
#include "../include/rdmaOps.hpp"
#include "../include/tensorOps.hpp"
#include "../include/logger.hpp"
#include "../include/attacks.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <lyra/lyra.hpp>
#include <string>
#include <thread>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <set>
#include <numeric>


using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

std::vector<torch::Tensor> run_fltrust_srvr(
  int n_clients,
  int rounds, 
  MnistTrain& mnist,
  int& srvr_ready_flag,
  float* srvr_w,
  std::vector<int>& clnt_ready_flags,
  std::vector<float*>& clnt_ws
);
std::vector<int> generateRandomUniqueVector(int n_clients, int min_sz = -1);
torch::Tensor aggregate_updates(
  const std::vector<torch::Tensor>& client_updates,
  const torch::Tensor& server_update
);

int main(int argc, char* argv[]) {
  Logger::instance().log("Server starting execution\n");
  int n_clients;

  std::string srvr_ip;
  std::string port;
  unsigned int posted_wqes;
  AddrInfo addr_info;
  std::shared_ptr<ltncyVec> latency = std::make_shared<ltncyVec>();
  latency->reserve(10);
  auto cli = lyra::cli() |
    lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
    lyra::opt(port, "port")["-p"]["--port"]("port") | 
    lyra::opt(n_clients, "n_clients")["-w"]["--n_clients"]("n_clients");
  auto result = cli.parse({ argc, argv });
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
      << std::endl;
    return 1;
  }

  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());
  std::cout << "Server: n_clients = " << n_clients << "\n";
  std::cout << "Server: srvr_ip = " << srvr_ip << "\n";
  std::cout << "Server: port = " << port << "\n";

  std::vector<RegInfo> reg_info(n_clients);
  std::vector<RcConn> conns(n_clients);
  std::vector<comm_info> conn_data;
  std::vector<LocalInfo> loc_info(n_clients);

  // Data structures for server and clients
  int srvr_ready_flag = 0;
  float* srvr_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
  std::vector<int> clnt_ready_flags(n_clients, 0);
  std::vector<float*> clnt_ws(n_clients);
  for (int i = 0; i < n_clients; i++) {
    clnt_ws[i] = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
  }

  // memory registration
  for (int i = 0; i < n_clients; i++) {
    reg_info[i].addr_locs.push_back(castI(&srvr_ready_flag));           
    reg_info[i].addr_locs.push_back(castI(srvr_w));
    reg_info[i].addr_locs.push_back(castI(&clnt_ready_flags[i]));
    reg_info[i].addr_locs.push_back(castI(clnt_ws[i]));
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(REG_SZ_DATA);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(REG_SZ_DATA);
    // // reg_info[i].data_sizes.push_back(CAS_SIZE);
    reg_info[i].permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    // connect to clients
    conns[i].acceptConn(addr_info, reg_info[i]);
    conn_data.push_back(conns[i].getConnData());
    std::cout << "\nConnected to client " << i << "\n";
  }

  // Create a dummy set of weights, needed for first call to runMNISTTrain():
  //std::vector<torch::Tensor> w = runMnistTrain(w_dummy);
  MnistTrain mnist(100);
  std::vector<torch::Tensor> w = run_fltrust_srvr(
    GLOBAL_ITERS,
    n_clients,
    mnist,
    srvr_ready_flag,
    srvr_w,
    clnt_ready_flags,
    clnt_ws
  );

  {
    std::ostringstream oss;
    oss << "\nFINAL W:\n";
    oss << "  " << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 10)) << " ";
    Logger::instance().log(oss.str());
  }

  mnist.testModel();

  // free memory
  free(srvr_w);
  for (int i = 0; i < n_clients; i++) {
    free(clnt_ws[i]);
  }
  for(RcConn conn : conns) {
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

std::vector<torch::Tensor> run_fltrust_srvr(
  int rounds, 
  int n_clients,
  MnistTrain& mnist,
  int& srvr_ready_flag,
  float* srvr_w,
  std::vector<int>& clnt_ready_flags,
  std::vector<float*>& clnt_ws) {

  std::vector<torch::Tensor> w = mnist.testOG();
  printTensorSlices(w, 0, 5);
  Logger::instance().log("\nInitial run of minstrain done\n");

  for (int round = 1; round <= rounds; round++) {
    auto all_tensors = flatten_tensor_vector(w);
    std::vector<int> polled_clients = generateRandomUniqueVector(n_clients);
    std::vector<torch::Tensor> clnt_updates(polled_clients.size());

    // Copy to shared memory
    size_t total_bytes = all_tensors.numel() * sizeof(float);
    float* global_w = all_tensors.data_ptr<float>();
    std::memcpy(srvr_w, global_w, total_bytes);

    // Set the flag to indicate that the weights are ready for the clients to read
    srvr_ready_flag = round;

    // Run local training
    std::vector<torch::Tensor> g = mnist.runMnistTrain(w);

    //NOTE: RIGHT NOW EVERY CLIENT TRAINS AND READS THE AGGREGATED W IN EACH ROUND, 
    //BUT SRVR ONLY READS FROM A RANDOM SUBSET OF CLIENTS
    Logger::instance().log("polled_clients size: " + std::to_string(polled_clients.size()) + "\n");
    for (size_t i = 0; i < polled_clients.size(); i++) {
      int client = polled_clients[i];
      Logger::instance().log("reading flags from client: " + std::to_string(client) + "\n");
      while(clnt_ready_flags[client] != round) { 
        // Active waiting wasting resources, could be improved
      }

      size_t numel_server = REG_SZ_DATA / sizeof(float);
      torch::Tensor flat_tensor = torch::from_blob(
          clnt_ws[client], 
          {static_cast<long>(numel_server)}, 
          torch::kFloat32
      ).clone();

      //std::vector<torch::Tensor> clnt_w_vec = reconstruct_tensor_vector(flat_tensor, w);
      clnt_updates[client] = flat_tensor;
    }

    // {
    //   std::ostringstream oss;
    //   oss << "Server read gradients from clients:" << "\n";
    //   for(torch::Tensor clnt_g : clnt_updates) {
    //     oss << "\n  Client g:\n";
    //     oss << "    " << clnt_g.slice(0, 0, std::min<size_t>(clnt_g.numel(), 5)) << " ";
    //   }
    //   Logger::instance().log(oss.str());
    // }

    // Use attacks to simulate Byzantine clients
    //clnt_updates = no_byz(clnt_updates, mnist.getModel(), GLOBAL_LEARN_RATE, N_BYZ_CLNTS, mnist.getDevice());
    clnt_updates = krum_attack(
      clnt_updates, 
      mnist.getModel(), 
      GLOBAL_LEARN_RATE, 
      N_BYZ_CLNTS, 
      mnist.getDevice()
    );

    Logger::instance().log("Server: Done with Byzantine attack\n");

    // AGGREGATION PHASE //////////////////////
    // Update w for the next round
    torch::Tensor flat_srvr_update = flatten_tensor_vector(g);
    torch::Tensor aggregated_update = aggregate_updates(clnt_updates, flat_srvr_update);
    std::vector<torch::Tensor> aggregated_update_vec = reconstruct_tensor_vector(aggregated_update, w);
    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] - GLOBAL_LEARN_RATE * aggregated_update_vec[i];
    }

  }

  return w;
}

std::vector<int> generateRandomUniqueVector(int n_clients, int min_sz) {
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

torch::Tensor aggregate_updates(
  const std::vector<torch::Tensor>& client_updates,
  const torch::Tensor& server_update) {
  
  // Compute cosine similarity between each client update and server update
  std::vector<float> trust_scores;
  for (const auto& flat_client_update : client_updates) {

      // Compute cosine similarity
      torch::Tensor dot_product = torch::dot(flat_client_update, server_update);
      float client_norm = torch::norm(flat_client_update, 2).item<float>();
      float server_norm = torch::norm(server_update, 2).item<float>();
      float cosine_sim = dot_product.item<float>() / (client_norm * server_norm + 1e-10);
      
      // Apply ReLU (max with 0)
      float trust_score = std::max(0.0f, cosine_sim);
      trust_scores.push_back(trust_score);
  }

  {
    std::ostringstream oss;
    oss << "\nTRUST SCORES:\n";
    for(int i = 0; i < trust_scores.size(); i++) {
      oss << "Client " << i << " score: " << trust_scores[i] << "\n";
    }
    Logger::instance().log(oss.str());
  }
  
  // Normalize trust scores
  float sum_trust = 0.0f;
  for (float score : trust_scores) {
      sum_trust += score;
  }
  
  std::vector<float> normalized_scores;
  if (sum_trust > 0) {
      for (float score : trust_scores) {
          normalized_scores.push_back(score / sum_trust);
      }
  } else {
      // If all scores are 0, use uniform weights
      float uniform_weight = 1.0f / trust_scores.size();
      for (size_t i = 0; i < trust_scores.size(); i++) {
          normalized_scores.push_back(uniform_weight);
      }
  }
  

  // Add scaled client updates to the aggregated update
  torch::Tensor aggregated_update = torch::zeros_like(server_update);
  for (size_t i = 0; i < client_updates.size(); i++) {
    float weight = normalized_scores[i];
    aggregated_update += client_updates[i] * weight;
  }
  
  return aggregated_update;
}

