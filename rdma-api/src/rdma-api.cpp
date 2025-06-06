#include "../include/rdma-api.hpp"
#include <chrono>
#include <deque>
#include <iostream>
#include <thread>
#include <utility>
#define NOW() std::chrono::steady_clock::now()

namespace norm {

std::chrono::nanoseconds::rep
get_dur(std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end) {
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  return elapsed.count();
}

int post_send(ibv_wr_opcode opcode, const std::deque<comm_info> &send_info,
              const NetFlags &net_flags, ltncyPtr latency,
              unsigned int &posted_wqes) {
  int ret = 0;
  unsigned int addr_upperBound = 0;
  struct ibv_send_wr *send_wr = new ibv_send_wr[send_info[0].wqe_depth];
  struct ibv_send_wr *send_failure;
  std::vector<std::vector<struct ibv_sge>> sge_lists;
  memset(&send_failure, 0, sizeof(send_failure));
  for (unsigned int indx = 0; indx < send_info[0].wqe_depth; indx++)
    memset(&send_wr[indx], 0, sizeof(send_wr[indx]));
  if (send_info[0].l_indices.empty() ||
      send_info[0].local_addresses.size() < send_info[0].l_indices.size())
    addr_upperBound = send_info[0].local_addresses.size();
  else
    addr_upperBound = send_info[0].l_indices.size();
  std::vector<uint64_t> local_addr_offs;
  if (send_info[0].local_addr_offs.empty()) {
    local_addr_offs.resize(addr_upperBound);
    std::fill(local_addr_offs.begin(), local_addr_offs.end(), 0);
  } else {
    local_addr_offs = send_info[0].local_addr_offs;
  }
  for (unsigned int info = 0; info < send_info.size(); info++) {
    if (local_addr_offs.size() == addr_upperBound) {
      std::vector<struct ibv_sge> sge_list;
      for (unsigned int indx = 0; indx < addr_upperBound; indx++) {
        struct ibv_sge sge;
        memset(&sge, 0, sizeof(sge));
        if (!send_info[0].l_indices.empty()) {
          sge.addr = (uint64_t)send_info[info]
                         .local_addresses[send_info[info].l_indices[indx]] +
                     local_addr_offs[indx];
          sge.lkey =
              (uint32_t)send_info[info].l_keys[send_info[info].l_indices[indx]];
        } else {
          sge.addr = (uint64_t)send_info[info].local_addresses[indx] +
                     local_addr_offs[indx];
          sge.lkey = (uint32_t)send_info[info].l_keys[indx];
        }
        sge.length = (uint32_t)send_info[info].lengths[indx];

        sge_list.push_back(sge);
        sge_lists.push_back(sge_list);
      }
    } else {
      ret = -1;
      return ret;
    }
  }
  for (unsigned int indx = 0; indx < send_info[0].wqe_depth; indx++) {
    if (send_info[0].qp_type == IBV_QPT_RC) {
      rc_comm_info rc_send_info =
          boost::get<rc_comm_info>(send_info[0].addr_data);
      if (opcode == IBV_WR_RDMA_READ || opcode == IBV_WR_RDMA_WRITE) {
        if (send_info[0].remote_addr == 0 && send_info[0].r_key == 0) {
          send_wr[indx].wr.rdma.rkey =
              rc_send_info.r_keys[send_info[0].r_index];
          send_wr[indx].wr.rdma.remote_addr =
              rc_send_info.remote_addresses[send_info[0].r_index] +
              send_info[0].remote_addr_off;
        } else {
          send_wr[indx].wr.rdma.rkey = send_info[0].r_key;
          send_wr[indx].wr.rdma.remote_addr = send_info[0].remote_addr;
        }
      } else if (opcode == IBV_WR_ATOMIC_FETCH_AND_ADD ||
                 opcode == IBV_WR_ATOMIC_CMP_AND_SWP) {
        if (send_info[0].remote_addr == 0 && send_info[0].r_key == 0) {
          send_wr[indx].wr.atomic.rkey =
              rc_send_info.r_keys[send_info[0].r_index];
          send_wr[indx].wr.atomic.remote_addr =
              rc_send_info.remote_addresses[send_info[0].r_index] +
              send_info[0].remote_addr_off;
        } else {
          send_wr[indx].wr.atomic.rkey = send_info[0].r_key;
          send_wr[indx].wr.atomic.remote_addr = send_info[0].remote_addr;
        }
        send_wr[indx].wr.atomic.compare_add = send_info[0].compare_add;
        if (opcode == IBV_WR_ATOMIC_CMP_AND_SWP)
          send_wr[indx].wr.atomic.swap = send_info[0].swap;
      }
    } else {
      ud_comm_info ud_send_info =
          boost::get<ud_comm_info>(send_info[0].addr_data);
      send_wr[indx].wr.ud.ah = ud_send_info.addr_handlers[send_info[0].r_index];
      send_wr[indx].wr.ud.remote_qpn =
          ud_send_info.remote_qpns[send_info[0].r_index];
      send_wr[indx].wr.ud.remote_qkey =
          ud_send_info.remote_qkeys[send_info[0].r_index];
    }
    if (sge_lists.size() == send_info[0].wqe_depth) {
      send_wr[indx].sg_list = sge_lists[indx].data();
      send_wr[indx].num_sge = sge_lists[indx].size();
    } else {
      send_wr[indx].sg_list = sge_lists[0].data();
      send_wr[indx].num_sge = sge_lists[0].size();
    }
    send_wr[indx].opcode = opcode;
    send_wr[indx].send_flags = net_flags.get();
  }
  for (unsigned int indx = 0; indx < send_info[0].wqe_depth - 1; indx++)
    send_wr[indx].next = &send_wr[indx + 1];

  send_wr[send_info[0].wqe_depth - 1].next = nullptr;
  if (net_flags.is_sync) {
    auto start = NOW();
    ret = ibv_post_send(send_info[0].qp, send_wr, &send_failure);
    ret = poll_cq(send_info[0].cq, send_info[0].wqe_depth);
    auto end = NOW();
    latency->push_back({send_info[0].wqe_depth, get_dur(start, end)});
    posted_wqes += send_info[0].wqe_depth;
  } else {
    int posted_wqes_ceil = 1000 - 2 * send_info[0].wqe_depth;
    if (posted_wqes >= posted_wqes_ceil) {
      // Poll the CQ to avoid overflow
      ret = poll_cq(send_info[0].cq, posted_wqes_ceil);
      posted_wqes -= posted_wqes_ceil;
    }
    ret = ibv_post_send(send_info[0].qp, send_wr, &send_failure);
    posted_wqes += send_info[0].wqe_depth;
  }
  if (ret) {
    std::cerr << "post send failed " << strerror(errno) << "\n";
    ret = -1;
    delete[] send_wr;
    exit(-1);
  }
  
  delete[] send_wr;
  return ret;
}
int post_recv(const comm_info &recv_info, const NetFlags &net_flags,
              ltncyPtr latency, unsigned int &posted_wqes) {
  int ret = 0;
  unsigned int addr_upperBound = 0;
  struct ibv_recv_wr *recv_wr = new ibv_recv_wr[recv_info.wqe_depth];
  struct ibv_recv_wr *recv_failure;
  std::vector<struct ibv_sge> sge_list{};
  memset(&recv_failure, 0, sizeof(recv_failure));
  for (unsigned int indx = 0; indx < recv_info.wqe_depth - 1; indx++) {
    memset(&recv_wr[indx], 0, sizeof(recv_wr[indx]));
    recv_wr[indx].next = &recv_wr[indx + 1];
  }
  recv_wr[recv_info.wqe_depth - 1].next = nullptr;

  if (recv_info.l_indices.empty())
    addr_upperBound = recv_info.local_addresses.size();
  else
    addr_upperBound = recv_info.l_indices.size();

  std::vector<uint64_t> local_addr_offs;
  if (recv_info.local_addr_offs.empty()) {
    local_addr_offs.resize(addr_upperBound);
    std::fill(local_addr_offs.begin(), local_addr_offs.end(), 0);
  } else
    local_addr_offs = recv_info.local_addr_offs;
  if (local_addr_offs.size() == addr_upperBound) {
    for (unsigned int indx = 0; indx < addr_upperBound; indx++) {
      struct ibv_sge sge;
      memset(&sge, 0, sizeof(sge));
      if (!recv_info.l_indices.empty()) {
        sge.addr =
            (uint64_t)recv_info.local_addresses[recv_info.l_indices[indx]] +
            local_addr_offs[indx];
        sge.lkey = (uint32_t)recv_info.l_keys[recv_info.l_indices[indx]];
      } else {
        sge.addr =
            (uint64_t)recv_info.local_addresses[indx] + local_addr_offs[indx];
        sge.lkey = (uint32_t)recv_info.l_keys[indx];
      }
      sge.length = (uint32_t)recv_info.lengths[indx];
      sge_list.push_back(sge);
    }
  } else {
    ret = -1;
    return ret;
  }
  for (unsigned int indx = 0; indx < recv_info.wqe_depth; indx++) {
    recv_wr[indx].sg_list = sge_list.data();
    recv_wr[indx].num_sge = sge_list.size();
  }
  for (unsigned int indx = 0; indx < recv_info.wqe_depth - 1; indx++)
    recv_wr[indx].next = &recv_wr[indx + 1];

  recv_wr[recv_info.wqe_depth - 1].next = nullptr;

  if (net_flags.is_sync) {
    auto start = NOW();
    ret = ibv_post_recv(recv_info.qp, recv_wr, &recv_failure);
    ret = poll_cq(recv_info.cq, recv_info.wqe_depth);
    auto end = NOW();
    latency->push_back({recv_info.wqe_depth, get_dur(start, end)});
  } else {
    int posted_wqes_ceil = 1000 - 2 * recv_info.wqe_depth;
    if (posted_wqes >= posted_wqes_ceil) {
      // Poll the CQ to avoid overflow
      ret = poll_cq(recv_info.cq, posted_wqes_ceil);
      posted_wqes -= posted_wqes_ceil;
    }
    ret = ibv_post_recv(recv_info.qp, recv_wr, &recv_failure);
  }
  posted_wqes += recv_info.wqe_depth;
  if (ret) {}
    ret = -1;

  delete[] recv_wr;
  return ret;
}
int poll_cq(ibv_cq *cq, unsigned int poll_no) {
  int ret = 0;
  struct ibv_wc *wc = new ibv_wc[poll_no];
  unsigned int cur_polled = 0;
  while (cur_polled < poll_no) {
    if (cq) {
      cur_polled += ibv_poll_cq(cq, poll_no, wc);
    }
  }

  if (wc->status != IBV_WC_SUCCESS) {
    std::cerr << "wc status = " << wc->status << std::endl;
    ret = -1;
  }

  delete[] wc;
  return ret;
}
int send(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
         const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
         ltncyPtr latency, unsigned int &posted_wqes) {
  int ret = 0;
  std::vector<uint32_t> payload;
  if (conn_data.local_addresses.size() == 1 && payload_sizes.size() > 1) {
    payload.push_back(payload_sizes[0]);
    for (unsigned int indx = 1; indx < payload_sizes.size(); indx++)
      payload[0] += payload_sizes[indx];
  } else {
    payload = payload_sizes;
  }

  std::deque<comm_info> send_infos;
  for (unsigned int info = 0; info < local_info.size(); info++) {
    comm_info send_info(conn_data.qp_type);
    send_info = conn_data;
    send_infos.push_back(send_info);
    send_infos[info].lengths = payload;
    send_infos[info].local_addr_offs = local_info[info].offs;
    send_infos[info].l_indices = local_info[info].indices;
  }
  ret = post_send(IBV_WR_SEND, send_infos, net_flags, latency, posted_wqes);
  return ret;
}
//template <typename VecPair_t>
int receive(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
            const LocalInfo &local_info, const NetFlags &net_flags,
            ltncyPtr latency, unsigned int &posted_wqes) {
  int ret = 0;
  comm_info recv_info(conn_data.qp_type);
  recv_info = conn_data;
  std::vector<uint32_t> payload;
  if (conn_data.local_addresses.size() == 1 && payload_sizes.size() > 1) {
    payload.push_back(payload_sizes[0]);
    for (unsigned int indx = 1; indx < payload_sizes.size(); indx++)
      payload[0] += payload_sizes[indx];
  } else {
    payload = payload_sizes;
  }
  recv_info.lengths = payload;
  recv_info.local_addr_offs = local_info.offs;
  recv_info.l_indices = local_info.indices;

  ret = post_recv(recv_info, net_flags, latency, posted_wqes);
  return ret;
}
int read(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
         const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
         const RemoteInfo &remote_info, ltncyPtr latency,
         unsigned int &posted_wqes) {
  int ret = 0;
  std::vector<uint32_t> payload;
  if (conn_data.local_addresses.size() == 1 && payload_sizes.size() > 1) {
    payload.push_back(payload_sizes[0]);
    for (unsigned int indx = 1; indx < payload_sizes.size(); indx++)
      payload[0] += payload_sizes[indx];
  } else {
    payload = payload_sizes;
  }
  std::deque<comm_info> send_infos;
  for (unsigned int info = 0; info < local_info.size(); info++) {
    comm_info send_info(conn_data.qp_type);
    send_info = conn_data;
    send_infos.push_back(send_info);
    send_infos[info].lengths = payload;
    send_infos[info].local_addr_offs = local_info[info].offs;
    send_infos[info].l_indices = local_info[info].indices;
    send_infos[info].remote_addr_off = remote_info.off;
    send_infos[info].r_index = remote_info.indx;
    send_infos[info].remote_addr = remote_info.addr;
    send_infos[info].r_key = remote_info.r_key;
  }
  ret =
      post_send(IBV_WR_RDMA_READ, send_infos, net_flags, latency, posted_wqes);
  return ret;
}
int write(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
          const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
          const RemoteInfo &remote_info, ltncyPtr latency,
          unsigned int &posted_wqes) {
  int ret = 0;
  std::vector<uint32_t> payload;
  if (conn_data.local_addresses.size() == 1 && payload_sizes.size() > 1) {
    payload.push_back(payload_sizes[0]);
    for (unsigned int indx = 1; indx < payload_sizes.size(); indx++)
      payload[0] += payload_sizes[indx];
  } else {
    payload = payload_sizes;
  }

  std::deque<comm_info> send_infos;
  for (unsigned int info = 0; info < local_info.size(); info++) {
    comm_info send_info(conn_data.qp_type);
    send_info = conn_data;
    send_infos.push_back(send_info);
    send_infos[info].lengths = payload;
    send_infos[info].local_addr_offs = local_info[info].offs;
    send_infos[info].l_indices = local_info[info].indices;
    send_infos[info].remote_addr_off = remote_info.off;
    send_infos[info].r_index = remote_info.indx;
    send_infos[info].remote_addr = remote_info.addr;
    send_infos[info].r_key = remote_info.r_key;
  }

  ret =
      post_send(IBV_WR_RDMA_WRITE, send_infos, net_flags, latency, posted_wqes);

  return ret;
}
int FAA(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
        const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
        const RemoteInfo &remote_info, ltncyPtr latency,
        unsigned int &posted_wqes) {
  int ret = 0;
  std::vector<uint32_t> payload;
  if (conn_data.local_addresses.size() == 1 && payload_sizes.size() > 1) {
    payload.push_back(payload_sizes[0]);
    for (unsigned int indx = 1; indx < payload_sizes.size(); indx++)
      payload[0] += payload_sizes[indx];
  } else {
    payload = payload_sizes;
  }

  std::deque<comm_info> send_infos;
  for (unsigned int info = 0; info < local_info.size(); info++) {
    comm_info send_info(conn_data.qp_type);
    send_info = conn_data;
    send_infos.push_back(send_info);
    send_infos[info].lengths = payload;
    send_infos[info].local_addr_offs = local_info[info].offs;
    send_infos[info].l_indices = local_info[info].indices;
    send_infos[info].remote_addr_off = remote_info.off;
    send_infos[info].r_index = remote_info.indx;
    send_infos[info].remote_addr = remote_info.addr;
    send_infos[info].r_key = remote_info.r_key;
    send_infos[info].compare_add = local_info[0].compare_add;
    send_infos[info].swap = local_info[0].swap;
  }

  ret = post_send(IBV_WR_ATOMIC_FETCH_AND_ADD, send_infos, net_flags, latency,
                  posted_wqes);

  return ret;
}

int CAS(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
        const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
        const RemoteInfo &remote_info, ltncyPtr latency,
        unsigned int &posted_wqes) {
  int ret = 0;
  std::vector<uint32_t> payload;
  if (conn_data.local_addresses.size() == 1 && payload_sizes.size() > 1) {
    payload.push_back(payload_sizes[0]);
    for (unsigned int indx = 1; indx < payload_sizes.size(); indx++)
      payload[0] += payload_sizes[indx];
  } else {
    payload = payload_sizes;
  }

  std::deque<comm_info> send_infos;
  for (unsigned int info = 0; info < local_info.size(); info++) {
    comm_info send_info(conn_data.qp_type);
    send_info = conn_data;
    send_infos.push_back(send_info);
    send_infos[info].lengths = payload;
    send_infos[info].local_addr_offs = local_info[info].offs;
    send_infos[info].l_indices = local_info[info].indices;
    send_infos[info].remote_addr_off = remote_info.off;
    send_infos[info].r_index = remote_info.indx;
    send_infos[info].remote_addr = remote_info.addr;
    send_infos[info].r_key = remote_info.r_key;
    send_infos[info].compare_add = local_info[0].compare_add;
    send_infos[info].swap = local_info[0].swap;
  }

  ret = post_send(IBV_WR_ATOMIC_CMP_AND_SWP, send_infos, net_flags, latency,
                  posted_wqes);

  return ret;
}
}; // namespace norm
