#include "../include/rdmaOps.hpp"
#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"

RdmaOps::RdmaOps(std::vector<comm_info> conn_data) : 
  conn_data(conn_data),
  latency(std::make_shared<ltncyVec>()),
  posted_wqes(0) {
    
  latency->reserve(10);
}

RdmaOps::~RdmaOps() {
  // Do nothing
}

int RdmaOps::exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::read(this->conn_data[conn_data_idx], {size}, {local_info}, NetFlags(), remote_info, this->latency, this->posted_wqes);
}

int RdmaOps::exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::write(this->conn_data[conn_data_idx], {size}, {local_info}, NetFlags(), remote_info, this->latency, this->posted_wqes);
}

int RdmaOps::exec_rdma_CAS(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, uint64_t compare_add, uint64_t swap, int conn_data_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  local_info.compare_add = compare_add;
  local_info.swap = swap;
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::CAS(this->conn_data[conn_data_idx], {size}, {local_info}, NetFlags(), remote_info, this->latency, this->posted_wqes);
}

int RdmaOps::exec_rdma_read(uint32_t size, uint32_t same_idx, int conn_data_idx) {
  return exec_rdma_read(size, same_idx, same_idx, conn_data_idx);
}

int RdmaOps::exec_rdma_write(uint32_t size, uint32_t same_idx, int conn_data_idx) {
  return exec_rdma_write(size, same_idx, same_idx, conn_data_idx);
}

int RdmaOps::exec_rdma_CAS(uint32_t size, uint32_t same_idx, uint64_t compare_add, uint64_t swap, int conn_data_idx) {
  return exec_rdma_CAS(size, same_idx, same_idx, compare_add, swap, conn_data_idx);
}
