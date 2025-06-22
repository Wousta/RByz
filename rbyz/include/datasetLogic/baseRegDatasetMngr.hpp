#pragma once

#include "datasetLogic/subsetSampler.hpp"
#include "nets/nnet.hpp"
#include "structs.hpp"

#include <ATen/core/TensorBody.h>
#include <cuda_runtime.h>
#include <future>
#include <vector>

class BaseRegDatasetMngr {
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

  BaseRegDatasetMngr(int worker_id, int num_workers, int64_t subset_size,
                     std::unique_ptr<NNet> net);
  virtual ~BaseRegDatasetMngr();

  virtual std::vector<torch::Tensor>
  runTraining(int round, const std::vector<torch::Tensor> &w) = 0;
  virtual void runInference(std::vector<torch::Tensor> &w) = 0;
  virtual void runTesting() = 0;
  virtual void runInference(const std::vector<torch::Tensor> &w) = 0;
  std::vector<size_t> getClientsSamplesCount(
      const std::unordered_map<int64_t, std::vector<size_t>> &label_to_indices);
  std::vector<torch::Tensor>
  updateModelParameters(const std::vector<torch::Tensor> &w);

  std::vector<torch::Tensor> getInitialWeights();
  torch::Device getDevice() { return device; }
  const NNet &getModel() const { return *model; }


  // Getters for specific sample data
  uint64_t getSampleOffset(size_t image_idx);
  void* getSample(size_t image_idx);
  uint32_t* getOriginalIndex(size_t image_idx);
  int64_t* getLabel(size_t image_idx);
  float* getImage(size_t image_idx);

protected:
  const char *kDataRoot = "./data";
  size_t forward_pass_size;
  size_t error_start;
  std::unique_ptr<NNet> model;
  torch::Device device;
  cudaStream_t memcpy_stream_A; // Stream for async memcpy
  cudaStream_t memcpy_stream_B; // Stream for async memcpy

  std::future<void> forward_pass_future;
  std::atomic<bool> pending_forward_pass{false};


  // Buffers for concurrent forward pass processing
  ForwardPassBuffer current_buffer;
  ForwardPassBuffer pending_buffer;

  // Buffer for runInference, which is not concurrent
  ForwardPassBuffer inference_buffer;

  void processForwardPassConcurrent(ForwardPassBuffer buffer);
  void processForwardPass(ForwardPassBuffer buffer);
  void processBatchResults(const torch::Tensor &output,
                           const torch::Tensor &targets,
                           const torch::Tensor &individual_losses,
                           size_t &curr_idx);


  virtual void buildLabelToIndicesMap() = 0;
  virtual void initDataInfo(const std::vector<size_t> &indices, int img_size);
  torch::Device init_device();
  SubsetSampler get_subset_sampler(
      int worker_id, size_t dataset_size, int64_t subset_size,
      const std::unordered_map<int64_t, std::vector<size_t>> &label_to_indices);
  std::vector<torch::Tensor> calculateUpdateCuda(const std::vector<torch::Tensor> &w_cuda);
  std::vector<torch::Tensor> calculateUpdateCPU(const std::vector<torch::Tensor> &w);

  template <typename DataLoader>
  void test(DataLoader &data_loader);

  template <typename DataLoader>
  void train(size_t epoch, torch::optim::Optimizer& optimizer, DataLoader &data_loader);

  template<typename DataLoader>
  void runInferenceBase(const std::vector<torch::Tensor>& w, DataLoader &registered_loader);

  // Registered dataset specific methods
  inline char* getBasePointerForIndex(size_t image_idx) const {
    return static_cast<char*>(data_info.reg_data) + (image_idx * data_info.get_sample_size());
  }
};