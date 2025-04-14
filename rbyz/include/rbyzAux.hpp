#pragma once

#include "../include/rdmaOps.hpp"
#include "../include/logger.hpp"
#include "../include/globalConstants.hpp"
#include <atomic>
#include <vector>
#include <thread>
#include <float.h>

// Forward declaration of ClientData struct (since it's defined in srvr.cpp)
struct ClientData {
    int clnt_index;
    float* updates;
    float* loss;
    float* error_rate;
};

void readClntsRByz(
    int n_clients,
    RdmaOps& rdma_ops,
    std::vector<std::atomic<int>>& clnt_CAS);

void updateTS(
    std::vector<ClientData>& clnt_data_vec,
    ClientData& clnt_data, 
    float srvr_loss, 
    float srvr_error_rate);