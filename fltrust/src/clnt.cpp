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

#define MSG_SZ 32
using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

int exec_rdma_op(int loc_info_idx, int rem_info_idx, uint64_t size, int op_type) {
  return 0;
}

int main(int argc, char* argv[]) {
  Logger::instance().log("Client starting execution\n");

  int id;
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
    lyra::opt(id, "id")["-p"]["--id"]("id");
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

  // Data structures for server and this client
  int srvr_ready_flag = 0;
  float* srvr_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
  int clnt_ready_flag = 0;
  float* clnt_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));

  // memory registration
  reg_info.addr_locs.push_back(castI(&srvr_ready_flag));
  reg_info.addr_locs.push_back(castI(srvr_w));
  reg_info.addr_locs.push_back(castI(&clnt_ready_flag));
  reg_info.addr_locs.push_back(castI(srvr_w));
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  // connect to server
  Logger::instance().log("Connecting to server\n");
  RcConn conn;
  int ret = conn.connect(addr_info, reg_info);
  comm_info conn_data = conn.getConnData();
  RdmaOps rdma_ops(conn_data);

  MnistTrain mnist;
  std::vector<torch::Tensor> w = mnist.testOG();
  //std::vector<torch::Tensor> w = runMnistTrain(w_dummy);
  printTensorSlices(w, 0, 10);

  for (int round = 1; round <= GLOBAL_ITERS; round++) {

    do {
      rdma_ops.exec_rdma_read(sizeof(int), SRVR_READY_IDX);
    } while (srvr_ready_flag != round);

    Logger::instance().log("Client read flag = " + std::to_string(srvr_ready_flag) + "\n");
    
    //std::this_thread::sleep_for(std::chrono::hours(1));

    // Read the weights from the server
    rdma_ops.exec_rdma_read(REG_SZ_DATA, SRVR_W_IDX);

    size_t numel_server = REG_SZ_DATA / sizeof(float);
    torch::Tensor flat_tensor = torch::from_blob(
        srvr_w, 
        {static_cast<long>(numel_server)}, 
        torch::kFloat32
    ).clone();

    w = reconstruct_tensor_vector(flat_tensor, w);

    // Print the first few updated weight values from server
    {
      std::ostringstream oss;
      oss << "Number of elements in updated tensor: " << flat_tensor.numel() << "\n";
      oss << "Updated weights from server:" << "\n";
      oss << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 10)) << " ";
      Logger::instance().log(oss.str());
    }

    // Run the training on the updated weights
    std::vector<torch::Tensor> g = mnist.runMnistTrain(w);

    // Send the updated weights back to the server
    auto all_tensors = flatten_tensor_vector(g);
    size_t total_bytes_g = all_tensors.numel() * sizeof(float);
    if(total_bytes_g != (size_t)REG_SZ_DATA) {
      Logger::instance().log("REG_SZ_DATA and total_bytes sent do not match!!\n");
    }
    float* client_w = all_tensors.data_ptr<float>();
    std::memcpy(castV(reg_info.addr_locs[CLNT_W_IDX]), client_w, total_bytes_g);
    unsigned int total_bytes_g_int = static_cast<unsigned int>(REG_SZ_DATA);
    rdma_ops.exec_rdma_write(total_bytes_g_int, CLNT_W_IDX);

    // Print the first few updated weights sent by client
    {
      std::ostringstream oss;
      oss << "Updated weights sent by client:" << "\n";
      oss << all_tensors.slice(0, 0, std::min<size_t>(all_tensors.numel(), 10)) << "\n";
      Logger::instance().log(oss.str());
    }

    // Update the ready flag
    clnt_ready_flag = round;
    rdma_ops.exec_rdma_write(sizeof(int), CLNT_READY_IDX);

    Logger::instance().log("Client: Done with iteration " + std::to_string(round) + "\n");

  }

  {
    std::ostringstream oss;
    oss << "\nFINAL W:\n";
    oss << "  " << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 20)) << " ";
    Logger::instance().log(oss.str());
  }

  free(srvr_w);
  free(clnt_w);
  free(addr_info.ipv4_addr);
  free(addr_info.port);
  conn.disconnect();

  std::cout << "Client done\n";

  return 0;
}
