#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "../include/mnistTrain.hpp"
#include "../include/globalConstants.hpp"
#include "../include/rdmaOps.hpp"
#include "../include/tensorOps.hpp"
#include "../include/logger.hpp"

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


#define MSG_SZ 32
using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

std::vector<int> generateRandomUniqueVector(int n_clients);
void update_global_model(float* global_w, const torch::Tensor& update, size_t model_size);
std::vector<torch::Tensor> flTrustAggregation(
    const std::vector<torch::Tensor>& server_update,
    const std::vector<float*>& clnt_ws,
    const std::vector<int>& polled_clients,
    int n_clients,
    const std::vector<size_t>& tensor_sizes,
    size_t total_params);

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
  }



  // Create a dummy set of weights, needed for first call to runMNISTTrain():
  TensorOps tensor_ops;
  std::vector<torch::Tensor> w_dummy;
  w_dummy.push_back(torch::arange(0, 10, torch::kFloat32));
  //std::vector<torch::Tensor> w = runMnistTrain(w_dummy);
  std::vector<torch::Tensor> w = runMnistTrainDummy(w_dummy);
  tensor_ops.printTensorSlices(w, 0, 10);
  Logger::instance().log("\nInitial run of minstrain done\n");

  for (int round = 1; round <= GLOBAL_ITERS; round++) {

    // Store w in shared memory
    auto all_tensors = torch::cat(w).contiguous();
    size_t total_bytes = all_tensors.numel() * sizeof(float);
    // Extract the float pointer from all_tensors
    float* global_w = all_tensors.data_ptr<float>();

    // Now use this pointer with memcpy
    std::memcpy(srvr_w, global_w, total_bytes);

    //std::cout << "\nServer wrote bytes = " << total_bytes << "\n";
    Logger::instance().log("\nServer wrote bytes = " + std::to_string(total_bytes) + "\n");
    {
      std::ostringstream oss;
      oss << "Updated weights from server:" << "\n";
      oss << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 10)) << " ";
      oss << "...\n";
      Logger::instance().log(oss.str());
    }

    // Set the flag to indicate that the weights are ready for the clients to read
    srvr_ready_flag = round;

    // Run local training
    //std::vector<torch::Tensor> g = runMnistTrain(w);

    // Read the gradients from the clients
    //NOTE: RIGHT NOW SOME CLIENTS DO TRAINING, BUT EVERY CLIENT READS THE AGGREGATED W IN EACH ROUND
    // for (int client : polled_clients) {
    //   while(clnt_ready_flags[client] != round) { 
    //     // Active waiting wasting resources, could be improved
    //   }

    //   auto g_flat = torch::cat(g).contiguous();
    //   size_t total_bytes_g = g_flat.numel() * sizeof(float);
    //   std::memcpy(clnt_ws[client], g_flat.data_ptr<float>(), total_bytes_g);
    // }

    // int clnt_idx = 0;
    // while (clnt_idx != n_clients) {
    //   if(clnt_ready_flags[clnt_idx] == round) {
    //     auto g_flat = torch::cat(g).contiguous();
    //     size_t total_bytes_g = g_flat.numel() * sizeof(float);
    //     std::memcpy(clnt_ws[clnt_idx], g_flat.data_ptr<float>(), total_bytes_g);
    //     clnt_idx++;
    //   }
    // }

    // {
    //   std::ostringstream oss;
    //   oss << "Server read gradients from clients:" << "\n";
    //   for (int i = 0; i < n_clients; i++) {
    //     oss << "Client " << i << ":\n";
    //     oss << torch::from_blob(clnt_ws[i], { static_cast<long>(REG_SZ_DATA / sizeof(float)) }, torch::kFloat32).slice(0, 0, std::min<size_t>(REG_SZ_DATA / sizeof(float), 10)) << " ";
    //     oss << "...\n";
    //   }
    //   Logger::instance().log(oss.str());
    // }

    // // AGGREGATION PHASE //////////////////////

    // // Poll just some of the client updates
    // std::vector<int> polled_clients = generateRandomUniqueVector(n_clients);

    // std::vector<torch::Tensor> aggregated_update = aggregateResults(g, clnt_ws, polled_clients);

    // // Update w for the next round
    // w = aggregated_update;

  }

  // free memory
  free(srvr_w);
  for (int i = 0; i < n_clients; i++) {
    free(clnt_ws[i]);
  }
  free(addr_info.ipv4_addr);
  free(addr_info.port);

  std::cout << "Server done\n";

  // sleep for server to be available
  Logger::instance().log("Sleeping for 1 hour\n");

  std::this_thread::sleep_for(std::chrono::hours(1));
  return 0;
}

std::vector<int> generateRandomUniqueVector(int n_clients) {
  // Initialize random number generator
  std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
  
  // Create a vector with all possible values from 0 to n
  std::vector<int> allValues(n_clients + 1);
  for (int i = 0; i <= n_clients; i++) {
      allValues[i] = i;
  }
  
  // Shuffle the vector
  std::shuffle(allValues.begin(), allValues.end(), rng);
  
  // Generate random size (must be less than n+1 to ensure we have enough unique values)
  std::uniform_int_distribution<int> sizeDist(0, n_clients);
  int size = sizeDist(rng);
  
  // Return the first 'size' elements
  return std::vector<int>(allValues.begin(), allValues.begin() + size);
}

