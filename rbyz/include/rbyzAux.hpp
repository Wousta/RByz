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
    std::atomic<int> clnt_CAS;
    float trust_score;
    float* updates;
    float* loss;        // Single value
    float* error_rate;  // Single value

    // Dataset data used
    size_t images_mem_size;
    size_t labels_mem_size;
    std::vector<size_t> inserted_indices;  // Indices the server put a test into that might be in the forward pass table

    // Forward pass data used
    size_t forward_pass_mem_size;
    size_t forward_pass_indices_mem_size;
    float* forward_pass;
    uint32_t* forward_pass_indices;

    ~ClientDataRbyz() {
        // Only free memory that was allocated with malloc/new
        if (updates) free(updates);
        if (loss) free(loss);
        if (error_rate) free(error_rate);
        if (forward_pass_indices) free(forward_pass_indices);
    }
};

void aquireCASLock(
    int clnt_idx, 
    RdmaOps& rdma_ops,
    std::atomic<int> &clnt_CAS);

void releaseCASLock(
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

void writeErrorAndLoss(
  BaseMnistTrain& mnist,
  float* clnt_w);

void runRByzClient(
    std::vector<torch::Tensor>& w,
    std::atomic<int> &clnt_CAS,
    RegisteredMnistTrain& mnist,
    float* clnt_w,
    float* loss_and_err);

void runRByzServer(
    int n_clients,
    std::vector<torch::Tensor>& w,
    RegisteredMnistTrain& mnist,
    RdmaOps& rdma_ops,
    std::vector<ClientDataRbyz>& clnt_data_vec);