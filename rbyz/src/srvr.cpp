#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "../include/mnistTrain.hpp"
#include "../include/globalConstants.hpp"
#include "../include/rdmaOps.hpp"
#include "../include/tensorOps.hpp"
#include "../include/logger.hpp"
#include "../include/attacks.hpp"
#include "../include/rbyzAux.hpp"

#include <float.h>
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

struct RegMemSrvr {
  private:
  int n_clients;
  
  public:
  int srvr_ready_flag = 0;
  float* srvr_w = reinterpret_cast<float*>(malloc(REG_SZ_DATA));
  std::vector<int> clnt_ready_flags;
  std::vector<float*> clnt_ws;
  std::vector<std::atomic<int>> clnt_CAS;
  
  RegMemSrvr(int n_clients) : 
    n_clients(n_clients),
    clnt_ready_flags(n_clients, 0),
    clnt_ws(n_clients),
    clnt_CAS(n_clients) {}
    
  ~RegMemSrvr() {
    free(srvr_w);
    for (int i = 0; i < n_clients; i++) {
      free(clnt_ws[i]);
    }
  }
};

void readClntsRByz(
  int n_clients,
  RdmaOps& rdma_ops,
  std::vector<std::atomic<int>>& clnt_CAS
);

std::vector<torch::Tensor> run_fltrust_srvr(
  int n_clients,
  int rounds, 
  MnistTrain& mnist,
  RegMemSrvr& regMem,
  std::vector<ClientData>& clntsData
);

std::vector<int> generateRandomUniqueVector(int n_clients, int min_sz = -1);

torch::Tensor aggregate_updates(
  std::vector<ClientData>& clnts_data,
  const std::vector<int>& polled_clients,
  const std::vector<torch::Tensor>& client_updates,
  const torch::Tensor& server_update
);

int main(int argc, char* argv[]) {
  Logger::instance().log("Server starting\n");
  int n_clients;
  bool load_model = false;
  std::string model_file = "mnist_model_params.pt";
  std::string srvr_ip;
  std::string port;
  unsigned int posted_wqes;
  AddrInfo addr_info;
  std::shared_ptr<ltncyVec> latency = std::make_shared<ltncyVec>();
  latency->reserve(10);
  auto cli = lyra::cli() |
    lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
    lyra::opt(port, "port")["-p"]["--port"]("port") | 
    lyra::opt(n_clients, "n_clients")["-w"]["--n_clients"]("n_clients") |
    lyra::opt(load_model)["-l"]["--load"]("Load model from saved file") |
    lyra::opt(model_file, "model_file")["-f"]["--file"]("Model file path");

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
  RegMemSrvr regMem(n_clients);
  std::vector<ClientData> clnt_data_vec(n_clients);

  //Point clnt_ws buffer parts to the right place
  for (int i = 0; i < n_clients; i++) {
    regMem.clnt_ws[i] = reinterpret_cast<float*> (malloc(REG_SZ_DATA + 2 * sizeof(float)));

    clnt_data_vec[i].clnt_index = i;
    clnt_data_vec[i].updates = regMem.clnt_ws[i];
    clnt_data_vec[i].loss = regMem.clnt_ws[i] + REG_SZ_DATA / sizeof(float);
    clnt_data_vec[i].error_rate = regMem.clnt_ws[i] + REG_SZ_DATA / sizeof(float) + 1;
  }

  // memory registration
  for (int i = 0; i < n_clients; i++) {
    reg_info[i].addr_locs.push_back(castI(&regMem.srvr_ready_flag));           
    reg_info[i].addr_locs.push_back(castI(regMem.srvr_w));
    reg_info[i].addr_locs.push_back(castI(&regMem.clnt_ready_flags[i]));
    reg_info[i].addr_locs.push_back(castI(regMem.clnt_ws[i]));
    reg_info[i].addr_locs.push_back(castI(&regMem.clnt_CAS[i]));

    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(REG_SZ_DATA);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    reg_info[i].data_sizes.push_back(REG_SZ_CLNT);
    reg_info[i].data_sizes.push_back(MIN_SZ);
    
    reg_info[i].permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    // connect to clients
    conns[i].acceptConn(addr_info, reg_info[i]);
    conn_data.push_back(conns[i].getConnData());
    std::cout << "\nConnected to client " << i << "\n";
  }

  MnistTrain mnist(0, SRVR_SUBSET_SIZE);
  std::vector<torch::Tensor> w;

  if (load_model) {
    w = mnist.loadModelState(model_file);
    if (w.empty()) {
      Logger::instance().log("Failed to load model state. Running FLTrust instead.\n");
      std::cout << "Failed to load model state from file: " << model_file << "\n";
      load_model = false;
    } else {
      Logger::instance().log("Successfully loaded model from file.\n");
      std::cout << "Loaded model state from file: " << model_file << "\n";

      // Do one iteration of fltrust with ALL clients to initialize trust scores
      w = run_fltrust_srvr(
        1,
        n_clients,
        mnist,
        regMem,
        clnt_data_vec
      );
    }
  }
  
  if (!load_model) {
    w = run_fltrust_srvr(
      GLOBAL_ITERS,
      n_clients,
      mnist,
      regMem,
      clnt_data_vec
    );

    mnist.saveModelState(w, model_file);
  }

  // Global rounds of RByz
  Logger::instance().log("Starting RBYZ\n");
  RdmaOps rdma_ops(conn_data);
  for (int round = 1; round < GLOBAL_ITERS_RBYZ; round++) {

    // Read the data from the clients
    readClntsRByz(n_clients, rdma_ops, regMem.clnt_CAS);
    
    // For each client run N rounds of RByz
    for(int j = 0; j < n_clients; j++) {

      int n = 15;
      for(int i = 0; i < n; i++) {

        // Read parameters from client
        aquireCASLock(j, rdma_ops, regMem.clnt_CAS);
        Logger::instance().log("CAS LOCK AQUIRED\n");
        rdma_ops.exec_rdma_read(REG_SZ_DATA, CLNT_W_IDX);
        releaseCASLock(j, rdma_ops, regMem.clnt_CAS);
        Logger::instance().log("CAS LOCK RELEASED\n");

        if(i != 1) {
          // UpdateTS
        }

        // Byz detection

        if(i == n) {
          // Aggregate
        }
      }
    }
  }

  {
    std::ostringstream oss;
    oss << "\nFINAL W:\n";
    oss << "  " << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 10)) << " ";
    Logger::instance().log(oss.str());
  }

  mnist.testModel();

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
  RegMemSrvr& regMem,
  std::vector<ClientData>& clntsData) {

  std::vector<torch::Tensor> w = mnist.testOG();
  printTensorSlices(w, 0, 5);
  Logger::instance().log("\nInitial run of minstrain done\n");

  for (int round = 1; round <= rounds; round++) {
    auto all_tensors = flatten_tensor_vector(w);
    std::vector<int> polled_clients = generateRandomUniqueVector(n_clients);
    std::vector<torch::Tensor> clnt_updates;

    // Copy to shared memory
    size_t total_bytes = all_tensors.numel() * sizeof(float);
    float* global_w = all_tensors.data_ptr<float>();
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

    //NOTE: RIGHT NOW EVERY CLIENT TRAINS AND READS THE AGGREGATED W IN EACH ROUND, 
    //BUT SRVR ONLY READS FROM A RANDOM SUBSET OF CLIENTS
    Logger::instance().log("polled_clients size: " + std::to_string(polled_clients.size()) + "\n");
    {
      std::ostringstream oss;
      oss << "Clients polled: \n";
      for(int index : polled_clients) {
        oss << index << " ";
      }
      oss << "\n";
      Logger::instance().log(oss.str());
    }
    for (size_t i = 0; i < polled_clients.size(); i++) {
      int client = polled_clients[i];
      Logger::instance().log("reading flags from client: " + std::to_string(client) + "\n");
      while(regMem.clnt_ready_flags[client] != round) { 
        std::this_thread::yield();
      }

      size_t numel_server = REG_SZ_DATA / sizeof(float);
      torch::Tensor flat_tensor = torch::from_blob(
          regMem.clnt_ws[client], 
          {static_cast<long>(numel_server)}, 
          torch::kFloat32
      ).clone();

      clnt_updates.push_back(flat_tensor);
    }

    // Use attacks to simulate Byzantine clients
    clnt_updates = no_byz(clnt_updates, mnist.getModel(), GLOBAL_LEARN_RATE, N_BYZ_CLNTS, mnist.getDevice());
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
    torch::Tensor aggregated_update = aggregate_updates(clntsData, polled_clients, clnt_updates, flat_srvr_update);
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
  std::vector<ClientData>& clnts_data,
  const std::vector<int>& polled_clients,
  const std::vector<torch::Tensor>& client_updates,
  const torch::Tensor& server_update) {
  
  // Compute cosine similarity between each client update and server update
  std::vector<float> trust_scores;
  for(int i = 0; i < client_updates.size(); i++) {
    torch::Tensor dot_product = torch::dot(client_updates[i], server_update);
    float client_norm = torch::norm(client_updates[i], 2).item<float>();
    float server_norm = torch::norm(server_update, 2).item<float>();
    float cosine_sim = dot_product.item<float>() / (client_norm * server_norm + 1e-10);
    
    // Apply ReLU (max with 0)
    float trust_score = std::max(0.0f, cosine_sim);
    trust_scores.push_back(trust_score);

    int clnt_idx = polled_clients[i];
    clnts_data[clnt_idx].trust_score = trust_score;
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


