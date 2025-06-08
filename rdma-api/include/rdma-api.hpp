#pragma once
#include "../../shared/net.hpp"
#include "../../shared/util.hpp"
#include "rc_conn.hpp" 
#include <chrono>
#include <memory>
#include <utility>

using ltncyPtr =
    std::shared_ptr<std::vector<std::pair<int, std::chrono::nanoseconds::rep>>>;

namespace norm {
int send(RcConn &conn, const std::vector<uint32_t> &payload_sizes,
         const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
         ltncyPtr latency);
int receive(RcConn &conn, const std::vector<uint32_t> &payload_sizes,
         const LocalInfo &local_info, const NetFlags &net_flags,
         ltncyPtr latency);
int read(RcConn &conn, const std::vector<uint32_t> &payload_sizes,
         const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
         const RemoteInfo &remote_info, ltncyPtr latency);
int write(RcConn &conn, const std::vector<uint32_t> &payload_sizes,
          const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
          const RemoteInfo &remote_info, ltncyPtr latency);
int FAA(RcConn &conn, const std::vector<uint32_t> &payload_sizes,
        const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
        const RemoteInfo &remote_info, ltncyPtr latency);
int CAS(RcConn &conn, const std::vector<uint32_t> &payload_sizes,
        const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
        const RemoteInfo &remote_info, ltncyPtr latency);
int poll_cq(ibv_cq *cq, unsigned int poll_no);
}; // namespace norm
