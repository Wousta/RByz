#pragma once

#include "nets/nnet.hpp"

#include <cuda_runtime.h>
#include <vector>

class BaseRegDatasetMngr {
public:
  BaseRegDatasetMngr(int worker_id, int num_workers, int64_t subset_size,
                     std::unique_ptr<NNet> net);
  virtual ~BaseRegDatasetMngr();

  // Should it return a reference or a copy?
  virtual const NNet &getModel() const { return *model; }

  template <typename DataLoader>
  void test(DataLoader &data_loader, size_t dataset_size);

  virtual torch::Device init_device();
  virtual std::vector<torch::Tensor> getInitialWeights();

protected:
  const char *kDataRoot = "./data";
  const int worker_id;
  const int num_workers;
  const int64_t subset_size;
  const int64_t kTrainBatchSize = 32;
  const int64_t kTestBatchSize = 1000;
  const int64_t kNumberOfEpochs = 2;
  const int64_t kLogInterval = 10;
  size_t test_dataset_size;
  float loss;
  float test_loss;
  float error_rate;
  float train_accuracy = 0.0; // Initialize train accuracy
  float test_accuracy = 0.0;  // Initialize test accuracy
  std::unordered_map<int64_t, std::vector<size_t>> label_to_indices;

  std::unique_ptr<NNet> model;
  torch::Device device;
  cudaStream_t memcpy_stream_A; // Stream for async memcpy
  cudaStream_t memcpy_stream_B; // Stream for async memcpy
};