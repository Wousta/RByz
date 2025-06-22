#include "datasetLogic/iRegDatasetMngr.hpp"
#include "datasetLogic/regMnistMngr.hpp"
#include "rc_conn.hpp"
#include "util.hpp"
#include "rdmaOps.hpp"
#include "tensorOps.hpp"
#include "global/globalConstants.hpp"
#include "rbyzAux.hpp"
#include "entities.hpp"
#include "nets/mnistNet.hpp"
#include "nets/cifar10Net.hpp"

//#include <logger.hpp>
#include <lyra/lyra.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

void registerClntMemory(RegInfo& reg_info, RegMemClnt& regMem, IRegDatasetMngr& mngr) {
  reg_info.addr_locs.push_back(castI(&regMem.srvr_ready_flag));
  reg_info.addr_locs.push_back(castI(regMem.srvr_w));
  reg_info.addr_locs.push_back(castI(&regMem.clnt_ready_flag));
  reg_info.addr_locs.push_back(castI(regMem.clnt_w));
  reg_info.addr_locs.push_back(castI(regMem.loss_and_err));
  reg_info.addr_locs.push_back(castI(&regMem.CAS));
  reg_info.addr_locs.push_back(castI(&regMem.local_step));
  reg_info.addr_locs.push_back(castI(&regMem.round));
  reg_info.addr_locs.push_back(castI(mngr.data_info.reg_data));
  reg_info.addr_locs.push_back(castI(mngr.f_pass_data.forward_pass));
  reg_info.addr_locs.push_back(castI(mngr.f_pass_data.forward_pass_indices));

  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(mngr.data_info.reg_data_size);
  reg_info.data_sizes.push_back(mngr.f_pass_data.forward_pass_mem_size);
  reg_info.data_sizes.push_back(mngr.f_pass_data.forward_pass_indices_mem_size);

  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
}

std::vector<torch::Tensor> run_fltrust_clnt(int rounds,
                                            RdmaOps& rdma_ops,
                                            IRegDatasetMngr& mngr,
                                            RegMemClnt& regMem) {

  std::vector<torch::Tensor> w = mngr.getInitialWeights();
  Logger::instance().log("Client: Initial run of minstrain done\n");

  for (int round = 1; round <= rounds; round++) {
    do {
      rdma_ops.exec_rdma_read(sizeof(int), SRVR_READY_IDX);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (regMem.srvr_ready_flag != round);

    Logger::instance().log("Client: Starting iteration " + std::to_string(round) + "\n");

    // Read the weights from the server and run training
    rdma_ops.read_mnist_update(w, regMem.srvr_w, SRVR_W_IDX);
    std::vector<torch::Tensor> g = mngr.runTraining(round, w);

    Logger::instance().log("Weight updates:\n");
    printTensorSlices(g, 0, 5);

    // Send the updated weights back to the server
    torch::Tensor all_tensors = flatten_tensor_vector(g);
    size_t total_bytes_g = all_tensors.numel() * sizeof(float);
    if(total_bytes_g != (size_t)REG_SZ_DATA) {
      throw std::runtime_error("REG_SZ_DATA and total_bytes sent do not match!!");
    }
    float* all_tensors_float = all_tensors.data_ptr<float>();
    std::memcpy(regMem.clnt_w, all_tensors_float, total_bytes_g);
    unsigned int total_bytes_g_int = static_cast<unsigned int>(REG_SZ_DATA);
    rdma_ops.exec_rdma_write(total_bytes_g_int, CLNT_W_IDX);

    // Update the ready flag
    regMem.clnt_ready_flag = round;
    rdma_ops.exec_rdma_write(sizeof(int), CLNT_READY_IDX);

    Logger::instance().log("Client: Done with iteration " + std::to_string(round) + "\n");
  }

  mngr.updateModelParameters(w);

  return w;
}

int main(int argc, char* argv[]) {
  Logger::instance().log("Client starting execution\n");

  int id;
  int n_clients;
  bool load_model = false;
  std::string model_file = "mnist_model_params.pt";
  std::string srvr_ip;
  std::string port;
  unsigned int posted_wqes;
  AddrInfo addr_info;
  RegInfo reg_info;
  std::shared_ptr<ltncyVec> latency = std::make_shared<ltncyVec>();
  latency->reserve(10);
  auto cli = lyra::cli() |
    lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
    lyra::opt(port, "port")["-p"]["--port"]("port") |
    lyra::opt(id, "id")["-p"]["--id"]("id") |
    lyra::opt(load_model)["-l"]["--load"]("Load model from saved file") |
    lyra::opt(n_clients, "n_clients")["-w"]["--n_clients"]("n_clients") |
    lyra::opt(model_file, "model_file")["-f"]["--file"]("Model file path");
  auto result = cli.parse({ argc, argv });
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
      << std::endl;
    return 1;
  }

  Logger::instance().log("Client: id = " + std::to_string(id) + "\n");
  Logger::instance().log("Client: srvr_ip = " + srvr_ip + "\n");
  Logger::instance().log("Client: port = " + port + "\n");
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());

  // Sleep to not overload the server when all clients connect
  std::this_thread::sleep_for(std::chrono::milliseconds(id * 800));

  MnistNet mnist_net;
  Cifar10Net cifar_net;

  std::unique_ptr<RegMnistMngr> reg_mnist_mngr =
    std::make_unique<RegMnistMngr>(id, n_clients + 1, CLNT_SUBSET_SIZE, mnist_net);

  // Struct to hold the registered data
  RegMemClnt regMem(id);
  registerClntMemory(reg_info, regMem, *reg_mnist_mngr);

  // connect to server
  RcConn conn;
  int ret = conn.connect(addr_info, reg_info);

  if (ret != 0) {
    throw std::runtime_error("Failed to connect to server");
  }

  std::vector<RcConn> conns = {conn}; 
  RdmaOps rdma_ops(conns);
  std:: cout << "\nClient id: " << id << " connected to server\n";
  Logger::instance().log("Client id: " + std::to_string(id) + " connected to server\n");

  Logger::instance().log("PRE: first mnist samples\n");
  for (int i = 0; i < 5; i++) {
    Logger::instance().log("Sample " + std::to_string(i) + ": label = " + std::to_string(*reg_mnist_mngr->getLabel(i)) + 
                           " | og_idx = " + std::to_string(*reg_mnist_mngr->getOriginalIndex(i)) + "\n");
  }

  std::vector<torch::Tensor> w = run_fltrust_clnt(GLOBAL_ITERS_FL, rdma_ops, *reg_mnist_mngr, regMem);
  // Label flipping for Byzantine clients
  // if (id <= N_BYZ_CLNTS) {
  //   Logger::instance().log("Client " + std::to_string(id) + " is Byzantine, flipping labels\n");
  //   std::mt19937 rng(42);
  //   registered_mnist->flipLabelsRandom(0.15f, rng);           // Flip 15% randomly
  //   registered_mnist->flipLabelsTargeted(7, 1, 0.30f, rng);   // Flip 30% of 7s to 1s
  // } 

  // Run the RByz client
  RByzAux rbyz_aux(rdma_ops, *reg_mnist_mngr);
  rbyz_aux.runRByzClient(w, regMem);

  std::cout << "\nClient done\n";
  std::this_thread::sleep_for(std::chrono::minutes(1));

  free(addr_info.ipv4_addr);
  free(addr_info.port);
  conn.disconnect();

  return 0;
}