#include "rdmaOps.hpp"
#include "rc_conn.hpp"
#include "rdma-api.hpp"
#include "util.hpp"
#include "global/logger.hpp"

RdmaOps::RdmaOps(std::vector<comm_info> conn_data) : 
  conn_data(conn_data),
  latency(std::make_shared<ltncyVec>()),
  posted_wqes(0) {
    
  latency->reserve(10);
  netflags_sync.send_flags = IBV_SEND_SIGNALED;
  netflags_sync.is_sync = true;
}

RdmaOps::~RdmaOps() {
  // Do nothing
}

int RdmaOps::exec_rdma_read(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::read(this->conn_data[conn_data_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency, this->posted_wqes);
}

int RdmaOps::exec_rdma_write(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, int conn_data_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::write(this->conn_data[conn_data_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency, this->posted_wqes);
}

int RdmaOps::exec_rdma_write(uint32_t size, LocalInfo &local_info, RemoteInfo &remote_info, int conn_data_idx, bool is_sync) {
  if (is_sync) {
    return norm::write(this->conn_data[conn_data_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency, this->posted_wqes);
  } else {
    return norm::write(this->conn_data[conn_data_idx], {size}, {local_info}, netflags_no_sync, remote_info, this->latency, this->posted_wqes);
  }
}

int RdmaOps::exec_rdma_CAS(uint32_t size, uint32_t loc_info_idx, uint32_t rem_info_idx, uint64_t compare_add, uint64_t swap, int conn_data_idx) {
  LocalInfo local_info;
  local_info.indices.push_back(loc_info_idx);
  local_info.compare_add = compare_add;
  local_info.swap = swap;
  RemoteInfo remote_info;
  remote_info.indx = rem_info_idx;
  return norm::CAS(this->conn_data[conn_data_idx], {size}, {local_info}, netflags_sync, remote_info, this->latency, this->posted_wqes);
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

void RdmaOps::read_mnist_update(std::vector<torch::Tensor> &update, float *local_w, int same_idx, int conn_data_idx) {
  // Read the update from the server
  Logger::instance().log("Client: Read weights from server DFA\n");
  if (conn_data_idx != 0) {
    exec_rdma_read(REG_SZ_DATA, same_idx, conn_data_idx);
  } else {
    exec_rdma_read(REG_SZ_DATA, same_idx);
  }

  size_t numel_server = REG_SZ_DATA / sizeof(float);
  Logger::instance().log("Client: Read weights from server numel = " + std::to_string(numel_server) + "\n");
  torch::Tensor flat_tensor = torch::from_blob(
      local_w, {static_cast<long>(numel_server)}, torch::kFloat32
  ).clone();
  Logger::instance().log("Client: Read weights from server done\n");
  update = reconstruct_tensor_vector(flat_tensor, update);
}
