#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "../include/rdmaOps.hpp"
#include "../include/tensorOps.hpp"
#include "../include/mnistTrain.hpp"
#include "../include/globalConstants.hpp"
#include "../include/logger.hpp"

//#include <logger.hpp>
#include <lyra/lyra.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

std::vector<torch::Tensor> run_fltrust_clnt(
  int rounds,
  RdmaOps& rdma_ops,
  MnistTrain& mnist,
  int& srvr_ready_flag,
  int& clnt_ready_flag,
  float* srvr_w,
  float* clnt_w
);

void writeErrorAndLoss(
  MnistTrain& mnist,
  float* clnt_w
);

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

  // addr
  Logger::instance().log("Client: id = " + std::to_string(id) + "\n");
  Logger::instance().log("Client: srvr_ip = " + srvr_ip + "\n");
  Logger::instance().log("Client: port = " + port + "\n");
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(id * 700));


  // Data structures for server and this client
  int srvr_ready_flag = 0;
  float* srvr_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
  int clnt_ready_flag = 0;
  float* clnt_w = reinterpret_cast<float*> (malloc(REG_SZ_CLNT));
  float* loss_and_err = reinterpret_cast<float*> (malloc(MIN_SZ));
  std::atomic<int> clnt_CAS(MEM_OCCUPIED);

  // memory registration
  reg_info.addr_locs.push_back(castI(&srvr_ready_flag));
  reg_info.addr_locs.push_back(castI(srvr_w));
  reg_info.addr_locs.push_back(castI(&clnt_ready_flag));
  reg_info.addr_locs.push_back(castI(clnt_w));
  reg_info.addr_locs.push_back(castI(loss_and_err));
  reg_info.addr_locs.push_back(castI(&clnt_CAS));
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  // connect to server
  RcConn conn;
  int ret = conn.connect(addr_info, reg_info);
  comm_info conn_data = conn.getConnData();
  RdmaOps rdma_ops({conn_data});
  std:: cout << "\nClient id: " << id << " connected to server ret: " << ret << "\n";

  MnistTrain mnist(id, n_clients + 1, CLNT_SUBSET_SIZE);
  std::vector<torch::Tensor> w;
  if (load_model) {
    w = mnist.loadModelState(model_file);
    if (w.empty()) {
      Logger::instance().log("Failed to load model state. Running FLTrust instead.\n");
      load_model = false;
    } else {
      Logger::instance().log("Successfully loaded model from file.\n");
      printTensorSlices(w, 0, 5);

      // Do one iteration of fltrust with one iteration to initialize trust scores
      std::cout << "CLNT Running FLTrust with loaded model\n";
      w = run_fltrust_clnt(
        1,
        rdma_ops,
        mnist,
        srvr_ready_flag,
        clnt_ready_flag,
        srvr_w,
        clnt_w
      );
      std::cout << "\nCLNT FLTrust with loaded model done\n";
    }
  }
  
  if (!load_model) {
    w = run_fltrust_clnt(
      GLOBAL_ITERS,
      rdma_ops,
      mnist,
      srvr_ready_flag,
      clnt_ready_flag,
      srvr_w,
      clnt_w
    );

    //mnist.saveModelState(w, model_file);
  }

  // Before rbyz, the client has to write error and loss for the first time
  writeErrorAndLoss(mnist, clnt_w);
  Logger::instance().log("Client: Initial loss and error values\n");
  clnt_CAS.store(MEM_FREE);

  // RBYZ client
  Logger::instance().log("\n\n=============================================\n");
  Logger::instance().log("==============  STARTING RBYZ  ==============\n");
  Logger::instance().log("=============================================\n");

  for (int round = 1; round < GLOBAL_ITERS_RBYZ; round++) {
    w = mnist.runMnistTrain(round, w, true);

    // Store the updated weights in clnt_w
    torch::Tensor all_tensors = flatten_tensor_vector(w);
    size_t total_bytes_g = all_tensors.numel() * sizeof(float);
    if(total_bytes_g != (size_t)REG_SZ_DATA) {
      Logger::instance().log("REG_SZ_DATA and total_bytes sent do not match!!\n");
    }

    float* all_tensors_float = all_tensors.data_ptr<float>();

    // Make server wait until memory is written
    int expected = MEM_FREE;
    while(!clnt_CAS.compare_exchange_strong(expected, MEM_OCCUPIED)) {
      std::this_thread::yield();
    }
    Logger::instance().log("CAS LOCK AQUIRED\n");

    // Store the updates, error and loss values in clnt_w
    std::memcpy(clnt_w, all_tensors_float, total_bytes_g);
    writeErrorAndLoss(mnist, loss_and_err);

    // Reset the memory ready flag
    clnt_CAS.store(MEM_FREE);
    Logger::instance().log("CAS LOCK RELEASED\n");
  }

  Logger::instance().log("Client: Final weights\n");
  printTensorSlices(w, 0, 5);

  free(srvr_w);
  free(clnt_w);
  free(loss_and_err);
  free(addr_info.ipv4_addr);
  free(addr_info.port);
  conn.disconnect();

  std::cout << "\nClient done\n";

  return 0;
}

std::vector<torch::Tensor> run_fltrust_clnt(
  int rounds,
  RdmaOps& rdma_ops,
  MnistTrain& mnist,
  int& srvr_ready_flag,
  int& clnt_ready_flag,
  float* srvr_w,
  float* clnt_w) {

    std::vector<torch::Tensor> w = mnist.getInitialWeights();
  Logger::instance().log("Client: Initial run of minstrain done\n");

  for (int round = 1; round <= rounds; round++) {
    do {
      rdma_ops.exec_rdma_read(sizeof(int), SRVR_READY_IDX);
    } while (srvr_ready_flag != round);

    Logger::instance().log("Client: Starting iteration " + std::to_string(round) + "\n");

    // Read the weights from the server
    rdma_ops.exec_rdma_read(REG_SZ_DATA, SRVR_W_IDX);

    size_t numel_server = REG_SZ_DATA / sizeof(float);
    torch::Tensor flat_tensor = torch::from_blob(
        srvr_w, 
        {static_cast<long>(numel_server)}, 
        torch::kFloat32
    ).clone();

    w = reconstruct_tensor_vector(flat_tensor, w);

    Logger::instance().log("Client: Read weights from server numel = " + std::to_string(flat_tensor.numel()) + "\n");

    // Run the training on the updated weights
    std::vector<torch::Tensor> g = mnist.runMnistTrain(round, w);

    // Keep updated values to follow FLtrust logic
    for (size_t i = 0; i < g.size(); ++i) {
      g[i] -= w[i];
    }

    Logger::instance().log("Weight updates:\n");
    printTensorSlices(g, 0, 5);

    // Send the updated weights back to the server
    torch::Tensor all_tensors = flatten_tensor_vector(g);
    size_t total_bytes_g = all_tensors.numel() * sizeof(float);
    if(total_bytes_g != (size_t)REG_SZ_DATA) {
      Logger::instance().log("REG_SZ_DATA and total_bytes sent do not match!!\n");
    }
    float* all_tensors_float = all_tensors.data_ptr<float>();
    std::memcpy(clnt_w, all_tensors_float, total_bytes_g);
    unsigned int total_bytes_g_int = static_cast<unsigned int>(REG_SZ_DATA);
    rdma_ops.exec_rdma_write(total_bytes_g_int, CLNT_W_IDX);

    // Print the first few updated weights sent by client
    // {
    //   std::ostringstream oss;
    //   oss << "Updated weights sent by client:" << "\n";
    //   oss << all_tensors.slice(0, 0, std::min<size_t>(all_tensors.numel(), 10)) << "\n";
    //   Logger::instance().log(oss.str());
    // }

    // Update the ready flag
    clnt_ready_flag = round;
    rdma_ops.exec_rdma_write(sizeof(int), CLNT_READY_IDX);

    Logger::instance().log("Client: Done with iteration " + std::to_string(round) + "\n");

  }

  return w;
}

void writeErrorAndLoss(MnistTrain& mnist, float* loss_and_err) {
  float loss_val = mnist.getLoss();
  float error_rate_val = mnist.getErrorRate();
  std::memcpy(loss_and_err, &loss_val, sizeof(float));
  std::memcpy(loss_and_err + 1, &error_rate_val, sizeof(float));
}
