#pragma once
#include "structs.hpp"
#include <torch/torch.h>
#include <random>
#include <vector>

class IRegDatasetMngr {
public:
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
  RegTrainData data_info;
  ForwardPassData f_pass_data;

  IRegDatasetMngr(int worker_id, int num_workers, int64_t subset_size)
      : worker_id(worker_id), num_workers(num_workers),
        subset_size(subset_size) {}
  virtual ~IRegDatasetMngr() = default;

  virtual std::vector<torch::Tensor>
  runTraining(int round, const std::vector<torch::Tensor> &w) = 0;
  virtual void runTesting() = 0;
  virtual void runInference(const std::vector<torch::Tensor> &w) = 0;

  virtual std::vector<torch::Tensor> getInitialWeights() = 0;
  virtual std::vector<torch::Tensor>
  updateModelParameters(const std::vector<torch::Tensor> &w) = 0;
  virtual torch::Device getDevice() = 0;
  virtual std::vector<size_t> getClientsSamplesCount() = 0;

  // Sample access methods
  virtual uint64_t getSampleOffset(size_t image_idx) = 0;
  virtual void *getSample(size_t image_idx) = 0;
  virtual uint32_t *getOriginalIndex(size_t image_idx) = 0;
  virtual int64_t *getLabel(size_t image_idx) = 0;
  virtual float *getImage(size_t image_idx) = 0;

  // Label flipping attacks
  virtual void flipLabelsRandom(float flip_ratio, std::mt19937 &rng) = 0;
  virtual void flipLabelsTargeted(int source_label, int target_label, float flip_ratio,
                          std::mt19937 &rng) = 0;
  virtual std::vector<size_t> findSamplesWithLabel(int label) = 0;
};