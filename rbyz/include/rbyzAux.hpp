#pragma once

#include "rdmaOps.hpp"
#include "datasetLogic/baseMnistTrain.hpp"
#include "datasetLogic/registeredMnistTrain.hpp"
#include "datasetLogic/regMnistSplitter.hpp"
#include "logger.hpp"
#include "global/globalConstants.hpp"
#include "entities.hpp"
#include <atomic>
#include <vector>
#include <thread>
#include <float.h>

void aquireClntCASLock(
    int clnt_idx, 
    RdmaOps& rdma_ops,
    std::atomic<int> &clnt_CAS);

void releaseClntCASLock(
    int clnt_idx, 
    RdmaOps& rdma_ops,
    std::atomic<int> &clnt_CAS);

void readClntsRByz(
    int n_clients,
    RdmaOps& rdma_ops,
    std::vector<ClientDataRbyz> &clnt_data_vec);

void updateTS(
    std::vector<ClientDataRbyz>& clnt_data_vec,
    ClientDataRbyz& clnt_data, 
    float srvr_loss, 
    float srvr_error_rate);

torch::Tensor aggregate_updates(
    const std::vector<torch::Tensor>& client_updates,
    const torch::Tensor& w,
    std::vector<ClientDataRbyz> &clnt_data_vec);

void writeErrorAndLoss(
  BaseMnistTrain& mnist,
  float* clnt_w);

void writeServerVD(
    void* vd_sample,
    RegMnistSplitter& splitter, 
    RegisteredMnistTrain& mnist,
    RdmaOps& rdma_ops,
    std::vector<ClientDataRbyz>& clnt_data);

bool compareVDOut(RegisteredMnistTrain& mnist, ClientDataRbyz& clnt_data);

void runRByzClient(
    std::vector<torch::Tensor>& w,
    RegisteredMnistTrain& mnist,
    RegMemClnt& regMem,
    RdmaOps& rdma_ops);

void runRByzServer(
    int n_clients,
    std::vector<torch::Tensor>& w,
    RegisteredMnistTrain& mnist,
    RdmaOps& rdma_ops,
    RegMemSrvr& regMem,
    std::vector<ClientDataRbyz>& clnt_data_vec);