#include "include/rdmaOps.hpp"
#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"

RdmaOps::RdmaOps(comm_info conn_data) 
    : conn_data(conn_data),
      latency(std::make_shared<ltncyVec>()),
      posted_wqes(0) {

    this->latency->reserve(10);
}

RdmaOps::~RdmaOps() {
  // Do nothing
}

int RdmaOps::exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::read(this->conn_data, {size}, {local_info}, NetFlags(), remote_info, this->latency, this->posted_wqes);
}

int RdmaOps::exec_rdma_read(uint32_t size, uint32_t same_idx) {
  return exec_rdma_read(size, same_idx, same_idx);
}

int RdmaOps::exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::write(this->conn_data, {size}, {local_info}, NetFlags(), remote_info, this->latency, this->posted_wqes);
}

int RdmaOps::exec_rdma_write(uint32_t size, uint32_t same_idx) {
  return exec_rdma_write(size, same_idx, same_idx);
}
