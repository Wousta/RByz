#pragma once
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstdint>
#include <infiniband/ib.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <string>
#include <vector>

// This file includes all struct that the user need to use the api

struct NetFlags {
  //unsigned int send_flags = IBV_SEND_SIGNALED;
  unsigned int send_flags; // Do not specify if is_sync is false
  bool is_sync = true; // Set this to false 
  unsigned int get() const { return send_flags; }
};
struct AddrInfo {
  char *ipv4_addr;
  char *port;
  unsigned int rdma_port = 0;
  bool operator==(const AddrInfo &other) const {
    return (ipv4_addr == other.ipv4_addr) && (port == other.port) &&
           (rdma_port == other.rdma_port);
  }
};
struct LocalInfo {
  std::vector<uint64_t> offs{};
  std::vector<uint64_t> indices{};
  int64_t compare_add = 0;  // expected
  int64_t swap = 0;         //desired
  uint64_t compare_add_mask = 0;
  uint64_t swap_mask = 0;
};
struct RemoteInfo {
  uint64_t indx = 0;
  uint64_t off = 0;
  uint64_t addr = 0;
  uint64_t r_key = 0;
};
struct RegInfo {
private:
  mutable ibv_pd *pd = nullptr;
  mutable ibv_cq *cq = nullptr;
  bool reg_dreg = false;
  friend class RcConn;

public:
  std::vector<std::uintptr_t> addr_locs{};
  std::vector<uint64_t> data_sizes{};
  int permissions;
  bool same_mr = false;
  int VL = -1;
};