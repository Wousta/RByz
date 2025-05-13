#pragma once

#include "rdmaOps.hpp"
#include "datasetLogic/baseMnistTrain.hpp"
#include "datasetLogic/registeredMnistTrain.hpp"
#include "global/logger.hpp"
#include "global/globalConstants.hpp"
#include <atomic>
#include <vector>
#include <thread>
#include <float.h>

// Used by the server to reference the registered data of each client
struct ClientDataRbyz {
    int clnt_index;
    float trust_score;
    float* updates;
    float* loss;
    float* error_rate;
};

void aquireCASLock(
    int clnt_idx, 
    RdmaOps& rdma_ops,
    std::vector<std::atomic<int>>& clnt_CAS);

void releaseCASLock(
    int clnt_idx, 
    RdmaOps& rdma_ops,
    std::vector<std::atomic<int>>& clnt_CAS);

void readClntsRByz(
    int n_clients,
    RdmaOps& rdma_ops,
    std::vector<std::atomic<int>>& clnt_CAS);

void updateTS(
    std::vector<ClientDataRbyz>& clnt_data_vec,
    ClientDataRbyz& clnt_data, 
    float srvr_loss, 
    float srvr_error_rate);

void writeErrorAndLoss(
  BaseMnistTrain& mnist,
  float* clnt_w);

void runRByzClient(
    std::vector<torch::Tensor>& w,
    std::atomic<int>& clnt_CAS,
    RegisteredMnistTrain& mnist,
    float* clnt_w,
    float* loss_and_err);