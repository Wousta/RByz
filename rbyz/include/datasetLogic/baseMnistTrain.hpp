#pragma once

#include "subsetSampler.hpp"
#include "global/globalConstants.hpp"
#include <vector>
#include <cuda_runtime.h>

struct Net : torch::nn::Module {
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.0, /*training=*/is_training()); // Dropout probability set to 0 for now during testing (was 0.5)
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

class BaseMnistTrain {
protected:
  const char* kDataRoot = "./data";
  const int worker_id;
  const int num_workers;
  const int64_t subset_size;
  const int64_t kTrainBatchSize = 32;
  const int64_t kTestBatchSize = 1000;
  const int64_t kNumberOfEpochs = 1;
  const int64_t kLogInterval = 10;
  torch::DeviceType device_type;
  torch::Device device;
  Net model;
  size_t test_dataset_size;
  torch::Tensor output;
  float loss;
  float error_rate;
  std::unordered_map<int64_t, std::vector<size_t>> label_to_indices; // 
  cudaStream_t memcpy_stream_A; // Stream for async memcpy
  cudaStream_t memcpy_stream_B; // Stream for async memcpy

  using DatasetType = decltype(
    torch::data::datasets::MNIST(kDataRoot)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>())
  );

  // Testing is done the same both registered and regular
  using TestDataLoaderType = torch::data::StatelessDataLoader<DatasetType, torch::data::samplers::RandomSampler>;
  DatasetType test_dataset;
  std::unique_ptr<TestDataLoaderType> test_loader;

  template <typename DataLoader>
  void test(
      Net& model,
      torch::Device device,
      DataLoader& data_loader,
      size_t dataset_size);

  torch::Device init_device();
  virtual SubsetSampler get_subset_sampler(int worker_id, size_t dataset_size, int64_t subset_size);

public:
  BaseMnistTrain(int worker_id, int num_workers, int64_t subset_size);
  virtual ~BaseMnistTrain();

  // Common interface methods
  virtual std::vector<torch::Tensor> runMnistTrain(int round, const std::vector<torch::Tensor>& w) = 0;
  virtual void runInference(const std::vector<torch::Tensor>& w) = 0;
  
  virtual std::vector<torch::Tensor> getInitialWeights();
  void saveModelState(const std::vector<torch::Tensor>& w, const std::string& filename);
  std::vector<torch::Tensor> loadModelState(const std::string& filename);
  void copyModelParameters(const Net& source_model);
  std::vector<torch::Tensor> updateModelParameters(const std::vector<torch::Tensor>& w);
  std::vector<size_t> getClientsSamplesCount();
  void buildLabelToIndicesMap();

  void testModel() {
    test(model, device, *test_loader, test_dataset_size);
  }

  // Getters and setters
  Net getModel() { return model; }
  torch::Device getDevice() { return device; }
  torch::Tensor getOutput() { return output; }
  float getLoss() { return loss; }
  void setLoss(float new_loss) { loss = new_loss; }
  float getErrorRate() { return error_rate; }
  void setErrorRate(float new_error_rate) { error_rate = new_error_rate; }
  int64_t getKTrainBatchSize() const { return kTrainBatchSize; }
};