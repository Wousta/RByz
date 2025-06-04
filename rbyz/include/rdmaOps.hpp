// Class for RDMA operations
#pragma once

#include "rc_conn.hpp"
#include "rdma-api.hpp"
#include "util.hpp"
#include "tensorOps.hpp"
#include "datasetLogic/registeredMnistTrain.hpp"

#include <vector>

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

class RdmaOps {
  private:
  std::vector<RcConn> conns;
  std::vector<comm_info> conn_data;
  std::shared_ptr<ltncyVec> latency;
  NetFlags netflags_sync;
  NetFlags netflags_no_sync;

  // Flow control thread members
  std::thread flow_control_thread;
  std::atomic<bool> stop_flow_control{false};
  std::condition_variable flow_control_cv;
  std::mutex flow_control_mutex;

  void flowControlWorker();

  public:
  RdmaOps(std::vector<RcConn>& conns);
  ~RdmaOps();

  void startFlowControl();
  void stopFlowControl();

  int exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx = 0);
  int exec_rdma_read(uint32_t size, uint32_t same_idx, int conn_data_idx = 0);
  
  int exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx = 0);
  int exec_rdma_write(uint32_t size, uint32_t same_idx, int conn_data_idx = 0);
  int exec_rdma_write(uint32_t size, LocalInfo &local_info, RemoteInfo &remote_info, int conn_data_idx = 0, bool is_sync = false);
  
  int exec_rdma_CAS(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, uint64_t compare_add, uint64_t swap, int conn_data_idx = 0);
  int exec_rdma_CAS(uint32_t size, uint32_t same_idx, uint64_t compare_add, uint64_t swap, int conn_data_idx = 0);

  void read_mnist_update(std::vector<torch::Tensor> &update, float *local_w, int same_idx, int conn_data_idx = 0);
};
