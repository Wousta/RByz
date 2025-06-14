// Class for RDMA operations
#pragma once

#include "rc_conn.hpp"
#include "rdma-api.hpp"
#include "util.hpp"

#include <vector>

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

class RdmaOps {
  private:
  std::vector<RcConn> conns;
  std::shared_ptr<ltncyVec> latency;

  public:
  RdmaOps(std::vector<RcConn> conns);
  ~RdmaOps();

  int exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_idx = 0);
  int exec_rdma_read(uint32_t size, uint32_t same_idx, int conn_idx = 0);
  
  int exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_idx = 0);
  int exec_rdma_write(uint32_t size, uint32_t same_idx, int conn_idx = 0);
  
  int exec_rdma_CAS(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, uint64_t compare_add, uint64_t swap, int conn_idx = 0);
  int exec_rdma_CAS(uint32_t size, uint32_t same_idx, uint64_t compare_add, uint64_t swap, int conn_idx = 0);
};
