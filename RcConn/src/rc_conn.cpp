#include "../include/rc_conn.hpp"
#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <numeric>
#include <poll.h>
#include <stdlib.h>
#include <string>
#include <thread>

int RcConn::assignContext(unsigned int device_num) {
  int ret = 0;
  int num_devices;
  struct ibv_device **device_list = ibv_get_device_list(&num_devices);
  if (device_list == nullptr) {
    ret = -1;
    std::cerr << "device_list is nullptr\n";
    exit(-1);
  }
  conn.context = ibv_open_device(device_list[device_num]);
  if (conn.context == nullptr) {
    ret = -1;
    std::cerr << "conn.context is nullptr\n";
    exit(-1);
  }
  ibv_free_device_list(device_list);
  return ret;
}
int RcConn::getLid() {
  struct ibv_port_attr port_attr;
  int port_num = 1;
  int ret = ibv_query_port(conn.context, 1, &port_attr);
  if (!ret) {
    return port_attr.lid;
  }
  return 0;
}
void RcConn::qpMkRdy(uint32_t qp_num, uint32_t dlid, uint8_t slvl,
                     int permissions) {
  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(qp_attr));
  qp_attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
  qp_attr.port_num = 1;
  qp_attr.pkey_index = 0;
  qp_attr.qp_access_flags = permissions;
  int ret = ibv_modify_qp(conn.qp, &qp_attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                              IBV_QP_ACCESS_FLAGS);
  if (ret) {
    std::cout << "failed to modify qp, error = " << strerror(errno) << "\n";
  }
  struct ibv_qp_attr rtr_attr;
  memset(&rtr_attr, 0, sizeof(rtr_attr));
  rtr_attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
  rtr_attr.path_mtu = ibv_mtu::IBV_MTU_4096;
  rtr_attr.rq_psn = 0;
  rtr_attr.max_dest_rd_atomic = 16;
  rtr_attr.min_rnr_timer = 12;
  rtr_attr.ah_attr.is_global = 0;
  rtr_attr.ah_attr.sl = slvl;
  rtr_attr.ah_attr.src_path_bits = 0;
  rtr_attr.ah_attr.port_num = 1;
  rtr_attr.dest_qp_num = qp_num;
  rtr_attr.ah_attr.dlid = dlid;
  ret = ibv_modify_qp(conn.qp, &rtr_attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret)
    std::cout << "failed to modify qp, error = " << strerror(errno) << "\n";
  struct ibv_qp_attr rts_attr;
  memset(&rts_attr, 0, sizeof(rts_attr));
  rts_attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
  rts_attr.timeout = 12;
  rts_attr.retry_cnt = 20;
  rts_attr.rnr_retry = 20;
  rts_attr.sq_psn = 0;
  rts_attr.max_rd_atomic = 16;
  ret = ibv_modify_qp(conn.qp, &rts_attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret)
    std::cout << "failed to modify qp, error = " << strerror(errno) << "\n";
}
int RcConn::queryParams() {
  int ret = 0;
  rdma_params.max_wr = 1000;
  rdma_params.max_cqe = 10000;
  rdma_params.max_sge = 30;
  rdma_params.max_qp_init_rd_atom = 8;
  rdma_params.max_res_rd_atom = 255;
  return ret;
}
int RcConn::createResAcc() {
  int ret = 0;
  conn.reg_info.pd = ibv_alloc_pd(conn.context);
  if (!conn.reg_info.pd)
    ret = -1;
  conn.cq = ibv_create_cq(conn.context, rdma_params.max_cqe, &conn, NULL, 0);
  if (conn.cq == nullptr)
    ret = -1;
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.cap.max_send_wr = rdma_params.max_wr;
  qp_init_attr.cap.max_recv_wr = rdma_params.max_wr;
  qp_init_attr.cap.max_send_sge = rdma_params.max_sge;
  qp_init_attr.cap.max_recv_sge = rdma_params.max_sge;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.send_cq = conn.cq;
  qp_init_attr.recv_cq = conn.cq;
  conn.qp = ibv_create_qp(conn.reg_info.pd, &qp_init_attr);
  for (unsigned int indx = 0; indx < conn.reg_info.addr_locs.size(); indx++) {
    conn.mr_list.push_back(
        ibv_reg_mr(conn.reg_info.pd, castV(conn.reg_info.addr_locs[indx]),
                   conn.reg_info.data_sizes[indx], conn.reg_info.permissions));
    conn.permissions = conn.reg_info.permissions;
    conn.app_addresses.push_back(castI(conn.mr_list[indx]->addr));
    conn.app_lengths.push_back(conn.mr_list[indx]->length);
    conn.l_keys.push_back(conn.mr_list[indx]->lkey);
  }
  return ret;
}
void RcConn::createMr() {
  for (unsigned int indx = 0; indx < conn.reg_info.data_sizes.size(); indx++) {
    conn.mr_list.push_back(
        ibv_reg_mr(conn.pd, castV(conn.reg_info.addr_locs[indx]),
                   conn.reg_info.data_sizes[indx], conn.reg_info.permissions));
    conn.permissions = conn.reg_info.permissions;
    conn.app_addresses.push_back(castI(conn.mr_list[indx]->addr));
    conn.app_lengths.push_back(conn.mr_list[indx]->length);
    conn.l_keys.push_back(conn.mr_list[indx]->lkey);
    conn.self_r_keys.push_back(conn.mr_list[indx]->rkey);
  }
}
int RcConn::createResInit() {
  int ret = 0;
  conn.pd = ibv_alloc_pd(conn.context);
  if (conn.pd == nullptr) {
    std::cerr << "pd is null ....exiting\n";
    exit(-1);
  }
  conn.cq = ibv_create_cq(conn.context, rdma_params.max_cqe, &conn, NULL, 0);
  if (conn.cq == nullptr) {
    std::cerr << "cq is null ....exiting\n";
    exit(-1);
  }
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.cap.max_send_wr = rdma_params.max_wr;
  qp_init_attr.cap.max_recv_wr = rdma_params.max_wr;
  qp_init_attr.cap.max_send_sge = rdma_params.max_sge;
  qp_init_attr.cap.max_recv_sge = rdma_params.max_sge;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.send_cq = conn.cq;
  qp_init_attr.recv_cq = conn.cq;
  conn.qp = ibv_create_qp(conn.pd, &qp_init_attr);
  createMr();
  return ret;
}
int RcConn::connectToSrvr() {
  unsigned int dlid, dst_qp_num;
  int lid = getLid();
  int nid = -1;
  if (lid != -1) {
    nid = std::stoi(GET("nid", conn.redis_context));
    while (GET("srvr", conn.redis_context) != "1") {
    }
    rtrvConnData(dlid, dst_qp_num);
    sendConnData(lid, conn.qp->qp_num);
    uint8_t sl = 0;
    if (conn.reg_info.VL != -1)
      sl = static_cast<uint8_t>(conn.reg_info.VL);
    qpMkRdy(dst_qp_num, dlid, sl, conn.reg_info.permissions);
  } else {
    std::cout << "WRONG LID\n";
  }
  SET("srvr", "0", conn.redis_context);
  RedisDcnct(conn.redis_context);
  return nid;
}
void RcConn::sendConnData(int lid, unsigned int qp_num) {
  std::string addrs;
  std::string rkeys;
  for (unsigned int indx = 0; indx < conn.mr_list.size(); indx++) {
    addrs += std::to_string(
        static_cast<uint64_t>((uintptr_t)conn.mr_list[indx]->addr));
    rkeys += std::to_string(static_cast<uint32_t>(conn.mr_list[indx]->rkey));
    if (indx != conn.mr_list.size() - 1) {
      addrs += ",";
      rkeys += ",";
    }
  }
  SET("lid", std::to_string(lid), conn.redis_context);
  SET("qp_num", std::to_string(qp_num), conn.redis_context);
  SET("addrs", addrs, conn.redis_context);
  SET("rkeys", rkeys, conn.redis_context);
  if (!conn.is_initiator) {
    SET("srvr", "1", conn.redis_context);
  } else {
    SET("clnt", "1", conn.redis_context);
  }
}
void RcConn::rtrvConnData(unsigned int &lid, unsigned int &qp_num) {
  lid = std::stoi(GET("lid", conn.redis_context));
  qp_num = std::stoi(GET("qp_num", conn.redis_context));
  std::stringstream ss(GET("addrs", conn.redis_context));
  std::string item;
  while (std::getline(ss, item, ',')) {
    conn.remote_addresses.push_back(std::stoull(item));
  }
  ss.clear();
  item.clear();
  ss.str(GET("rkeys", conn.redis_context));
  while (std::getline(ss, item, ',')) {
    try {
      conn.r_keys.push_back(std::stoul(item));
    } catch (const std::invalid_argument &) {
      std::cerr << "Invalid number: " << item << std::endl;
    }
  }
}
int RcConn::connectToClnt() {
  unsigned int dlid, dst_qp_num;
  int lid = getLid();
  int nid = -1;
  if (lid != -1) {
    nid = std::stoi(GET("nid", conn.redis_context));
    while (GET("srvr", conn.redis_context) != "0") {
    }
    sendConnData(lid, conn.qp->qp_num);
    while (GET("clnt", conn.redis_context) != "1") {
    }
    rtrvConnData(dlid, dst_qp_num);
    uint8_t sl = 0;
    if (conn.reg_info.VL != -1)
      sl = static_cast<uint8_t>(conn.reg_info.VL);
    qpMkRdy(dst_qp_num, dlid, sl, conn.reg_info.permissions);
  } else {
    std::cout << "WRONG LID\n";
  }
  SET("clnt", "0", conn.redis_context);
  RedisDcnct(conn.redis_context);
  return nid;
}
int RcConn::connectAcceptor() {
  int ret = 0;
  ret = assignContext(conn.addr_info.rdma_port);
  ret = createResAcc();
  ret = connectToClnt();
  return ret;
}
int RcConn::connectInitiator() {
  int ret = 0;
  ret = assignContext(conn.addr_info.rdma_port);
  ret = createResInit();
  ret = connectToSrvr();
  return ret;
}
int RcConn::clean() {
  int ret = 0;
  if (conn.qp)
    ibv_destroy_qp(conn.qp);
  if (conn.cq)
    ret = ibv_destroy_cq(conn.cq);
  if (ret)
    return ret;
  for (int indx = 0; indx < conn.mr_list.size(); indx++) {
    if (conn.mr_list[indx]) {
      ret = ibv_dereg_mr(conn.mr_list[indx]);
      if (ret)
        return ret;
    }
  }
  if (conn.is_initiator) {
    if (conn.pd)
      ret = ibv_dealloc_pd(conn.pd);
    else if (conn.reg_info.pd)
      ret = ibv_dealloc_pd(conn.reg_info.pd);
    ibv_close_device(conn.context);
  }
  return ret;
}
int RcConn::connect(const AddrInfo &addr_info, const RegInfo &reg_info) {
  int ret = 0;
  RedisCnct(conn.redis_context, addr_info.ipv4_addr, atoi(addr_info.port));
  conn.is_initiator = true;
  conn.addr_info = addr_info;
  conn.reg_info = reg_info;
  ret = queryParams();
  ret = connectInitiator();
  return ret;
}
int RcConn::acceptConn(const AddrInfo &addr_info, const RegInfo &reg_info) {
  int ret = 0;
  RedisCnct(conn.redis_context, addr_info.ipv4_addr, atoi(addr_info.port));
  conn.addr_info = addr_info;
  conn.reg_info = reg_info;
  ret = queryParams();
  ret = connectAcceptor();
  return ret;
}
comm_info RcConn::getConnData() {
  int ret = 0;
  struct comm_info info(IBV_QPT_RC);
  rc_comm_info &conn_data = boost::get<rc_comm_info>(info.addr_data);
  for (unsigned int i = 0; i < conn.app_addresses.size(); i++) {
    info.l_keys.push_back(conn.mr_list[i]->lkey);
    info.local_addresses.push_back(castI(conn.mr_list[i]->addr));
    info.lengths.push_back(conn.mr_list[i]->length);
    info.qp = conn.qp;
    info.cq = conn.cq;
  }
  conn_data.remote_addresses = conn.remote_addresses;
  conn_data.r_keys = conn.r_keys;
  conn_data.self_r_keys = conn.self_r_keys;
  info.app_addresses = conn.app_addresses;
  info.wqe_depth = conn.wqe_depth;
  return info;
}
int RcConn::disconnect() {
  int ret = 0;
  ret = clean();
  return ret;
}
