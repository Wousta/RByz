// Class for RDMA operations
#pragma once

#include "rc_conn.hpp"
#include "rdma-api.hpp"
#include "util.hpp"
#include "tensorOps.hpp"

#include <vector>

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

class RdmaOps {
  private:
  std::vector<comm_info> conn_data;
  std::shared_ptr<ltncyVec> latency;
  unsigned int posted_wqes;

  public:
  RdmaOps(std::vector<comm_info> conn_data);
  ~RdmaOps();

  int exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx = 0);
  int exec_rdma_read(uint32_t size, uint32_t same_idx, int conn_data_idx = 0);
  
  int exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx = 0);
  int exec_rdma_write(uint32_t size, uint32_t same_idx, int conn_data_idx = 0);
  
  int exec_rdma_CAS(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, uint64_t compare_add, uint64_t swap, int conn_data_idx = 0);
  int exec_rdma_CAS(uint32_t size, uint32_t same_idx, uint64_t compare_add, uint64_t swap, int conn_data_idx = 0);

  void read_mnist_update(std::vector<torch::Tensor> &update, float *local_w, int same_idx, int conn_data_idx = 0);
};
