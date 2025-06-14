#pragma once

#include "baseMnistTrain.hpp"
#include "registeredMNIST.hpp"
#include "structs.hpp"
#include <memory>
#include <future>
#include <atomic>

/**
 * @brief RegisteredMnistTrain class for handling registered MNIST dataset training.
 * 
 * Memory layout per sample:
 * [1 uint32_t (original index)][1 int64_t (label)][784 floats (pixels)]
 */
class RegisteredMnistTrain : public BaseMnistTrain {
private:  
  std::unique_ptr<RegisteredMNIST> registered_dataset;
  RegMnistTrainData data_info;
  ForwardPassData forward_pass_info;

  // Frequently accessed references extracted for convenience
  void* & reg_data = data_info.reg_data;
  size_t & num_samples = data_info.num_samples;
  size_t & reg_data_size = data_info.reg_data_size;
  const size_t & sample_size = data_info.sample_size;
  const size_t & index_size = data_info.index_size;
  const size_t & label_size = data_info.label_size;
  float* & forward_pass = forward_pass_info.forward_pass;
  uint32_t* & forward_pass_indices = forward_pass_info.forward_pass_indices;
  size_t & forward_pass_mem_size = forward_pass_info.forward_pass_mem_size;
  size_t & forward_pass_indices_mem_size = forward_pass_info.forward_pass_indices_mem_size;

  size_t forward_pass_size;
  size_t error_start;

  std::future<void> forward_pass_future;
  std::atomic<bool> pending_forward_pass{false};

  // Double buffer for forward pass data to avoid race conditions
  struct ForwardPassBuffer {
    std::vector<torch::Tensor> outputs;
    std::vector<torch::Tensor> targets; 
    std::vector<torch::Tensor> losses;
  };

  ForwardPassBuffer current_buffer;
  ForwardPassBuffer pending_buffer;

  void processForwardPassConcurrent(ForwardPassBuffer buffer);

  using RegTrainDataLoader = torch::data::StatelessDataLoader<RegisteredMNIST, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> registered_loader;

  void processBatchResults(
    const torch::Tensor& output, 
    const torch::Tensor& targets,
    const torch::Tensor& individual_losses,
    size_t& curr_idx);

  void processForwardPass(
      const std::vector<torch::Tensor>& outputs,
      const std::vector<torch::Tensor>& targets,
      const std::vector<torch::Tensor>& losses);
      
  void train(
      size_t epoch, 
      Net& model, 
      torch::Device device, 
      torch::optim::Optimizer& optimizer, 
      size_t dataset_size);

  // Helper method to get properly cast base pointer for a given image index
  char* getBasePointerForIndex(size_t image_idx) const {
    return static_cast<char*>(reg_data) + (image_idx * sample_size);
  }

public:
  RegisteredMnistTrain(int worker_id, int num_workers, int64_t subset_size);
  ~RegisteredMnistTrain();

  std::vector<torch::Tensor> runMnistTrain(int round, const std::vector<torch::Tensor>& w) override;
  void runInference(const std::vector<torch::Tensor>& w) override;
  
  // Getters for registered memory
  void* getRegisteredData() { return reg_data; }
  size_t getRegisteredDataSize() { return reg_data_size; }
  size_t getNumSamples() { return num_samples; }
  size_t getSampleSize() { return sample_size; }
  size_t getLabelSize() { return label_size; }
  float* getForwardPass() { return forward_pass; }
  uint32_t* getForwardPassIndices() { return forward_pass_indices; }
  size_t getForwardPassMemSize() { return forward_pass_mem_size; }
  size_t getForwardPassIndicesMemSize() { return forward_pass_indices_mem_size; }
  size_t getValuesPerSample() { return forward_pass_info.values_per_sample; }
  size_t getBytesPerValue() { return forward_pass_info.bytes_per_value; }

  uint64_t getSampleOffset(size_t image_idx) {
    if (image_idx >= num_samples) {
      throw std::out_of_range("Image index out of range in RegisteredMnistTrain::getSampleOffset()");
    }
    return image_idx * sample_size;
  }

  void* getSample(size_t image_idx) {
    if (image_idx >= num_samples) {
      throw std::out_of_range("Image index " + std::to_string(image_idx) + " out of range in RegisteredMnistTrain::getSample()");
    }

    return getBasePointerForIndex(image_idx);
  }

  uint32_t* getOriginalIndex(size_t image_idx) {
    return reinterpret_cast<uint32_t*>(getBasePointerForIndex(image_idx));
  }

  int64_t* getLabel(size_t image_idx) {
    return reinterpret_cast<int64_t*>(getBasePointerForIndex(image_idx) + index_size);
  }

  float* getImage(size_t image_idx) {
    return reinterpret_cast<float*>(getBasePointerForIndex(image_idx) + index_size + label_size);
  }

};