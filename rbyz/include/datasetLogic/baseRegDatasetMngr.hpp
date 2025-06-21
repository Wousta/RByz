#pragma once

#include "nets/nnet.hpp"

#include <cuda_runtime.h>
#include <vector>

class BaseRegDatasetMngr {
public:
  BaseRegDatasetMngr(int worker_id, int num_workers, int64_t subset_size,
                     std::unique_ptr<NNet> net);
  virtual ~BaseRegDatasetMngr();
  virtual std::vector<torch::Tensor> runTraining(int round, const std::vector<torch::Tensor>& w) = 0;
  virtual void runInference(const std::vector<torch::Tensor>& w) = 0;

  template <typename DataLoader>
  void test(DataLoader &data_loader, size_t dataset_size);

  float getTestLoss() { return test_loss; }
  void setTestLoss(float new_test_loss) { test_loss = new_test_loss; }
  float getLoss() { return loss; }
  void setLoss(float new_loss) { loss = new_loss; }
  float getErrorRate() { return error_rate; }
  float getTrainAccuracy() { return train_accuracy; }
  void setTrainAccuracy(float new_train_accuracy) { train_accuracy = new_train_accuracy; }
  float getTestAccuracy() { return test_accuracy; }
  void setTestAccuracy(float new_test_accuracy) { test_accuracy = new_test_accuracy; }
  void setErrorRate(float new_error_rate) { error_rate = new_error_rate; }
  int64_t getKTrainBatchSize() const { return kTrainBatchSize; }
  torch::Device getDevice() { return device; }
  const NNet &getModel() const { return *model; }

  std::vector<torch::Tensor> getInitialWeights();
  std::vector<torch::Tensor> updateModelParameters(const std::vector<torch::Tensor>& w);

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
  std::unique_ptr<NNet> model;
  torch::Device device;
  cudaStream_t memcpy_stream_A; // Stream for async memcpy
  cudaStream_t memcpy_stream_B; // Stream for async memcpy

  virtual torch::Device init_device();
};