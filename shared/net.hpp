#pragma once
#include "util.hpp"
#include <boost/variant.hpp>
#include <infiniband/ib.h>
#include <rdma/rdma_cma.h>
#include <vector>

struct ud_reg_info {
  std::vector<std::uintptr_t> addr_locs{};
  std::vector<uint32_t> data_sizes{};
  bool new_qp = true;
};
struct rc_comm_info {
  std::vector<uint64_t> remote_addresses{};
  std::vector<uint32_t> r_keys{};
  std::vector<uint32_t> self_r_keys{};
};
struct ud_comm_info {
  std::vector<struct ibv_ah *> addr_handlers{};
  std::vector<uint32_t> remote_qpns{};
  std::vector<uint32_t> remote_qkeys{};
};
struct comm_info {
  struct ibv_qp *qp = nullptr;
  struct ibv_cq *cq = nullptr;
  std::vector<uint64_t> l_keys{};
  std::vector<std::uintptr_t> app_addresses{};
  std::vector<std::uintptr_t> local_addresses{};
  std::vector<uint64_t> local_addr_offs{};
  std::vector<uint64_t> l_indices{};
  std::vector<uint32_t> lengths{};
  unsigned int r_index = 0;
  uint64_t remote_addr_off = 0;
  uint64_t wqe_depth = 1;
  uint64_t remote_addr = 0;
  uint64_t r_key = 0;
  int64_t compare_add = 0;
  int64_t swap = 0;
  uint64_t compare_add_mask = 0;
  uint64_t swap_mask = 0;
  enum ibv_qp_type qp_type;
  using new_var = boost::variant<rc_comm_info, ud_comm_info>;
  new_var addr_data;
  comm_info(enum ibv_qp_type type) : qp_type(type) {
    if (qp_type == IBV_QPT_RC)
      addr_data = rc_comm_info{};
    else
      addr_data = ud_comm_info{};
  }
};
