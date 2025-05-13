#pragma once

#include "baseMnistTrain.hpp"
#include "registeredMNIST.hpp"
#include <memory>

class RegisteredMnistTrain : public BaseMnistTrain {
private:  
  // Registered memory dataset components
  float* registered_images; 
  int64_t* registered_labels;
  size_t registered_samples; 
  std::unique_ptr<RegisteredMNIST> registered_dataset;
  
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
  size_t getRegisteredSamplesCount() { return registered_samples; }
};