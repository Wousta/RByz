// Class for RDMA operations

#ifndef RDMA_OPS_HPP
#define RDMA_OPS_HPP

#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"

#include <vector>

using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

class RdmaOps {
  private:
  comm_info conn_data;
  std::shared_ptr<ltncyVec> latency;
  unsigned int posted_wqes;

  public:
  RdmaOps(comm_info conn_data);
  ~RdmaOps();

  // Function to execute RDMA operation
  int exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx);
  int exec_rdma_read(uint32_t size, uint32_t same_idx);
  int exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx);
  int exec_rdma_write(uint32_t size, uint32_t same_idx);
};

#endif