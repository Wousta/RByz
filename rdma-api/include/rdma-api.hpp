#pragma once
#include "../../shared/net.hpp"
#include "../../shared/util.hpp"
#include <chrono>
#include <memory>
#include <utility>

using ltncyPtr =
    std::shared_ptr<std::vector<std::pair<int, std::chrono::nanoseconds::rep>>>;

namespace norm {
int send(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
         const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
         ltncyPtr latency, unsigned int &posted_wqes);
int receive(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
         const LocalInfo &local_info, const NetFlags &net_flags,
         ltncyPtr latency, unsigned int &posted_wqes);
int read(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
         const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
         const RemoteInfo &remote_info, ltncyPtr latency,
         unsigned int &posted_wqes);
int write(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
          const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
          const RemoteInfo &remote_info, ltncyPtr latency,
          unsigned int &posted_wqes);
int FAA(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
        const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
        const RemoteInfo &remote_info, ltncyPtr latency,
        unsigned int &posted_wqes);
int CAS(comm_info &conn_data, const std::vector<uint32_t> &payload_sizes,
        const std::vector<LocalInfo> &local_info, const NetFlags &net_flags,
        const RemoteInfo &remote_info, ltncyPtr latency,
        unsigned int &posted_wqes);
int poll_cq(ibv_cq *cq, unsigned int poll_no);
}; // namespace norm
