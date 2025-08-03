#include "rdmaOps.hpp"
#include "rc_conn.hpp"
#include "rdma-api.hpp"
#include "util.hpp"

RdmaOps::RdmaOps(std::vector<RcConn>& conns) : conns(conns), latency(std::make_shared<ltncyVec>()) {    
  latency->reserve(10);
  netflags_sync.is_sync = true;
  netflags_no_sync.is_sync = false;
}

RdmaOps::~RdmaOps() {
  // Do nothing
}

int RdmaOps::exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::read(conns[conn_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency);
}

int RdmaOps::exec_rdma_read(uint32_t size, LocalInfo &local_info, RemoteInfo &remote_info, int conn_idx, Mode mode) {
  if (mode == Mode::sync) {
    return norm::read(conns[conn_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency);
  } else {
    return norm::read(conns[conn_idx], {size}, {local_info}, netflags_no_sync, remote_info, this->latency);
  }
}

int RdmaOps::exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::write(conns[conn_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency);
}

int RdmaOps::exec_rdma_write(uint32_t size, LocalInfo &local_info, RemoteInfo &remote_info, int conn_idx, Mode mode) {
  if (mode == Mode::sync) {
    return norm::write(conns[conn_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency);
  } else {
    return norm::write(conns[conn_idx], {size}, {local_info}, netflags_no_sync, remote_info, this->latency);
  }
}

int RdmaOps::exec_rdma_CAS(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, uint64_t compare_add, uint64_t swap, int conn_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  local_info.compare_add = compare_add;
  local_info.swap = swap;
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::CAS(conns[conn_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency);
}

int RdmaOps::exec_rdma_read(uint32_t size, uint32_t same_idx, int conn_idx) {
  return exec_rdma_read(size, same_idx, same_idx, conn_idx);
}

int RdmaOps::exec_rdma_write(uint32_t size, uint32_t same_idx, int conn_idx) {
  return exec_rdma_write(size, same_idx, same_idx, conn_idx);
}

int RdmaOps::exec_rdma_CAS(uint32_t size, uint32_t same_idx, uint64_t compare_add, uint64_t swap, int conn_idx) {
  return exec_rdma_CAS(size, same_idx, same_idx, compare_add, swap, conn_idx);
}