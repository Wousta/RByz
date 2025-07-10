#pragma once

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "datasetLogic/subsetSampler.hpp"
#include "nets/cifar10Net.hpp"
#include "nets/mnistNet.hpp"
#include "structs.hpp"

#include <ATen/core/TensorBody.h>
#include <cuda_runtime.h>
#include <future>
#include <torch/data/datasets/base.h>
#include <utility>
#include <vector>

template <typename NetType> class BaseRegDatasetMngr : public IRegDatasetMngr {
public:
  BaseRegDatasetMngr(int worker_id, TrainInputParams &t_params, NetType net);
  virtual ~BaseRegDatasetMngr();

  inline std::vector<torch::Tensor>
  calculateUpdate(const std::vector<torch::Tensor> &w) override {
    if (device.is_cuda()) {
      return calculateUpdateCuda(w);
    } else {
      return calculateUpdateCPU(w);
    }
  }

  virtual void runTraining() override = 0;
  virtual void runTesting() override = 0;
  virtual void runInference() override = 0;
  virtual void renewDataset(float proportion = 1.0, std::optional<int> seed = std::nullopt) override = 0;

  std::vector<size_t> getClientsSamplesCount(uint32_t clnt_subset_size,
                                             uint32_t srvr_subset_size,
                                             uint32_t dataset_size) override;
  std::vector<torch::Tensor>
  updateModelParameters(const std::vector<torch::Tensor> &w) override;
  std::vector<torch::Tensor> getInitialWeights() override;
  torch::Device getDevice() override { return device; }
  const NetType &getModel() const { return model; }

  // Label flipping attacks
  void flipLabelsRandom(float flip_ratio, std::mt19937 &rng) override;
  void flipLabelsTargeted(int source_label, int target_label, float flip_ratio,
                          std::mt19937 &rng) override;
  void corruptImagesRandom(float flip_ratio, std::mt19937 &rng) override;
  std::vector<size_t> findSamplesWithLabel(int label) override;

  // Getters for specific sample data
  inline uint64_t getSampleOffset(size_t image_idx) override {
    if (image_idx >= data_info.num_samples) {
      throw std::out_of_range("Image index out of range in "
                              "RegisteredMnistTrain::getSampleOffset()");
    }
    return image_idx * data_info.get_sample_size();
  }

  inline void *getSample(size_t image_idx) override {
    if (image_idx >= data_info.num_samples) {
      throw std::out_of_range(
          "Image index " + std::to_string(image_idx) +
          " out of range in RegisteredMnistTrain::getSample()");
    }

    return getBasePointerForIndex(image_idx);
  }

  inline uint32_t *getOriginalIndex(size_t image_idx) override {
    return reinterpret_cast<uint32_t *>(getBasePointerForIndex(image_idx));
  }

  inline int64_t *getLabel(size_t image_idx) override {
    return reinterpret_cast<int64_t *>(getBasePointerForIndex(image_idx) +
                                       data_info.index_size);
  }

  // UINT8CHANGE
  // inline uint8_t *getImage(size_t image_idx) override {
  //   return reinterpret_cast<uint8_t *>(getBasePointerForIndex(image_idx) +
  //                                   data_info.index_size +
  //                                   data_info.label_size);
  // }
  inline float *getImage(size_t image_idx) override {
    return reinterpret_cast<float *>(getBasePointerForIndex(image_idx) +
                                     data_info.index_size +
                                     data_info.label_size);
  }

protected:
  uint64_t renew_idx = 0; // Index for build dataset when renewing samples for RByz
  std::map<int64_t, std::vector<size_t>> label_to_indices;
  std::vector<size_t> indices; // Indices of samples in the registered dataset
  size_t forward_pass_size;
  size_t error_start;
  NetType model;
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
      int worker_id, size_t dataset_size, int64_t subset_size, uint32_t srvr_subset_size,
      const std::map<int64_t, std::vector<size_t>> &label_to_indices);
  std::vector<torch::Tensor>
  calculateUpdateCuda(const std::vector<torch::Tensor> &w_cuda);
  std::vector<torch::Tensor>
  calculateUpdateCPU(const std::vector<torch::Tensor> &w);

  template <typename DataLoader> void test(DataLoader &data_loader);

  template <typename DataLoader>
  void train(size_t epoch, torch::optim::Optimizer &optimizer,
             DataLoader &data_loader);

  template <typename DataLoader>
  void runInferenceBase(DataLoader &registered_loader);

  // Registered dataset specific methods
  inline char *getBasePointerForIndex(size_t image_idx) const {
    return static_cast<char *>(data_info.reg_data) +
           (image_idx * data_info.get_sample_size());
  }
};

extern template class BaseRegDatasetMngr<MnistNet>;
extern template class BaseRegDatasetMngr<Cifar10Net>;