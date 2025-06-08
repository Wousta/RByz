#pragma once
#include "../../shared/macros.hpp"
#include "../../shared/net.hpp"
#include "../../shared/redis.hpp"
#include "../../shared/util.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <infiniband/ib.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <rdma/rdma_cma.h>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

class RcConn {
private:
  struct Conn {
    struct ibv_context *context = nullptr;
    struct redisContext *redis_context = nullptr;
    struct ibv_pd *pd = nullptr;
    struct ibv_qp *qp = nullptr;
    unsigned int posted = 0;
    unsigned int polled = 0;
    std::vector<std::uintptr_t> app_addresses{};
    std::vector<uint64_t> app_lengths{};
    unsigned int permissions;
    std::vector<struct ibv_mr *> mr_list{};
    struct ibv_cq *cq = nullptr;
    bool connected = false;
    std::vector<uint64_t> remote_addresses{};
    std::vector<uint32_t> r_keys{};
    std::vector<uint32_t> self_r_keys{};
    std::vector<uint32_t> l_keys{};
    uint32_t remote_mrs_no = 0;
    uint32_t wqe_depth = 1;
    bool is_initiator = false;
    std::vector<struct ibv_dm *> dm_list{};
    AddrInfo addr_info;
    RegInfo reg_info;
  };
  struct RdmaMaxParams {
    uint32_t max_wr = 0;
    uint32_t max_cqe = 0;
    uint32_t max_sge = 0;
    uint32_t max_qp_init_rd_atom = 0;
    uint32_t max_res_rd_atom = 0;
  };
  Conn conn;
  RdmaMaxParams rdma_params;
  std::vector<unsigned int> cores;

  int init();
  int queryParams();
  int assignContext(unsigned int device_num);
  int createResAcc();
  void createMr();
  int createResInit();
  void sendConnData(int lid, unsigned int qp_num);
  void rtrvConnData(unsigned int &lid, unsigned int &qp_num);
  int connectToClnt();
  int connectToSrvr();
  int connectAcceptor();
  int connectInitiator();
  int clean();
  void qpMkRdy(uint32_t qp_num, uint32_t dlid, uint8_t slvl, int permissions);
  int getLid();
  friend struct RegInfo;

public:
  int connect(const AddrInfo &addr_info, const RegInfo &reg_info);
  // if successful returns a virtual node id else returns -1
  int acceptConn(const AddrInfo &addr_info, const RegInfo &reg_info);
  int readDM(void *host_mem, uint64_t dm_idx, uint64_t dm_offset,
             size_t length);
  int writeDM(void *host_mem, uint64_t dm_idx, uint64_t dm_offset,
              size_t length);
  int disconnect();
  comm_info getConnData();
  unsigned int getPosted() { return conn.posted; }
  unsigned int getPolled() { return conn.polled; }
  void addPosted(unsigned int num) { conn.posted += num; }
  void addPolled(unsigned int num) { conn.polled += num; }
};
