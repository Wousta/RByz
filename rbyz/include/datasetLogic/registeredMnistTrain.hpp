#pragma once

#include "baseMnistTrain.hpp"
#include "registeredMNIST.hpp"
#include <memory>

class RegisteredMnistTrain : public BaseMnistTrain {
private:  
  // Memory layout:
  // For each image i:
  // - Pixels: registered_images[i * data_size] to registered_images[i * data_size + 783]
  // - Original index: reinterpret_cast<uint32_t*>(&registered_images[i * data_size + 784])
  float* registered_images; 
  int64_t* registered_labels;
  float* forward_pass; // CHANGE TO ERROR AND LOSS
  int32_t* forward_pass_indices;
  size_t images_mem_size;
  size_t labels_mem_size;
  size_t forward_pass_mem_size;
  size_t forward_pass_indices_mem_size;
  size_t registered_samples; 
  std::unique_ptr<RegisteredMNIST> registered_dataset;
  const size_t data_size = 785; // 784 pixels + 1 index
  
  using RegTrainDataLoader = torch::data::StatelessDataLoader<RegisteredMNIST, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> registered_loader;
      
  void train(
      size_t epoch, 
      Net& model, 
      torch::Device device, 
      torch::optim::Optimizer& optimizer, 
      size_t dataset_size);

public:
  RegisteredMnistTrain(int worker_id, int num_workers, int64_t subset_size);
  ~RegisteredMnistTrain();

  std::vector<torch::Tensor> runMnistTrain(int round, const std::vector<torch::Tensor>& w) override;
  void runInference() override;
  
  // Getters for registered memory
  float* getRegisteredImages() { return registered_images; }
  int64_t* getRegisteredLabels() { return registered_labels; }
  float* getForwardPass() { return forward_pass; }
  int32_t* getForwardPassIndices() { return forward_pass_indices; }
  size_t getRegisteredImagesMemSize() { return images_mem_size; }
  size_t getRegisteredLabelsMemSize() { return labels_mem_size; }
  size_t getForwardPassMemSize() { return forward_pass_mem_size; }
  size_t getForwardPassIndicesMemSize() { return forward_pass_indices_mem_size; }
  size_t getRegisteredSamplesCount() { return registered_samples; }
  size_t getDataSize() { return data_size; }
  
  float* getImagePixels(size_t image_idx) {
    return registered_images + (image_idx * data_size);
  }

  uint32_t getOriginalIndex(size_t image_idx) {
      uint32_t* index_ptr = reinterpret_cast<uint32_t*>(&registered_images[image_idx * data_size + 784]);
      return *index_ptr;
  }

  void setOriginalIndex(size_t image_idx, uint32_t original_idx) {
      uint32_t* index_ptr = reinterpret_cast<uint32_t*>(&registered_images[image_idx * data_size + 784]);
      *index_ptr = original_idx;
  }
};