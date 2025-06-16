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

class RByzAux {
private:
    RdmaOps& rdma_ops;
    RegisteredMnistTrain& mnist;

    void updateTS(
        std::vector<ClientDataRbyz>& clnt_data_vec,
        ClientDataRbyz& clnt_data, 
        float srvr_loss, 
        float srvr_error_rate);

    torch::Tensor aggregate_updates(
        const std::vector<torch::Tensor>& client_updates,
        const torch::Tensor& w,
        const std::vector<ClientDataRbyz>& clnt_data_vec,
        const std::vector<uint32_t>& clnt_indices);

    void writeServerVD(
        void* vd_sample,
        RegMnistSplitter& splitter, 
        std::vector<ClientDataRbyz>& clnt_data);

    bool processVDOut(ClientDataRbyz& clnt_data, bool check_byz);

public:
    RByzAux(RdmaOps& rdma_ops, RegisteredMnistTrain& mnist)
        : rdma_ops(rdma_ops), mnist(mnist) {}

    RByzAux() = delete;
    
    void runRByzClient(
        std::vector<torch::Tensor>& w,
        RegMemClnt& regMem);

    void runRByzServer(
        int n_clients,
        std::vector<torch::Tensor>& w,
        RegMemSrvr& regMem,
        std::vector<ClientDataRbyz>& clnt_data_vec);


};