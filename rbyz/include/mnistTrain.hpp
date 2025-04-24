#pragma once

#include "../include/subsetSampler.hpp"
#include "../include/globalConstants.hpp"

#include <vector>


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
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

class MnistTrain {
private:
  const char* kDataRoot = "./data";
  const int worker_id;
  const int64_t subset_size;
  const int64_t kTrainBatchSize = 64;
  const int64_t kTestBatchSize = 1;
  const int64_t kNumberOfEpochs = 10;
  const int64_t kLogInterval = 10;
  const float learnRate = GLOBAL_LEARN_RATE;
  torch::DeviceType device_type;
  torch::Device device;
  Net model;
  size_t train_dataset_size;
  size_t test_dataset_size;
  torch::Tensor output;
  float loss;
  float error_rate;
  
  using DatasetType = decltype(
    torch::data::datasets::MNIST(kDataRoot)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>())
  );

  using SubsetSamplerType = SubsetSampler;

  // Define the DataLoader type with sampler (for train_loader)
  using TrainDataLoaderType = torch::data::StatelessDataLoader<DatasetType, SubsetSamplerType>;
  
  // Define the DataLoader type without sampler (for test_loader)
  using TestDataLoaderType = torch::data::StatelessDataLoader<DatasetType, torch::data::samplers::RandomSampler>;

  DatasetType train_dataset;
  DatasetType test_dataset;
  std::unique_ptr<TrainDataLoaderType> train_loader;
  std::unique_ptr<TestDataLoaderType> test_loader;

  torch::Device init_device();
  SubsetSampler get_subset_sampler(int worker_id, size_t dataset_size, int64_t subset_size);

public:
  MnistTrain(
    int worker_id, 
    int64_t subset_size
  );
  ~MnistTrain() = default;

  template <typename DataLoader>
  void train(
      size_t epoch,
      Net& model,
      torch::Device device,
      DataLoader& data_loader,
      torch::optim::Optimizer& optimizer,
      size_t dataset_size);

  template <typename DataLoader>
  void test(
      Net& model,
      torch::Device device,
      DataLoader& data_loader,
      size_t dataset_size);

  std::vector<torch::Tensor> runMnistTrain(int round, const std::vector<torch::Tensor>& w);
  std::vector<torch::Tensor> testOG();
  void testModel();
  void runInference();
  Net getModel() { return model; }
  torch::Device getDevice() { return device; }
  torch::Tensor getOutput() { return output; }
  void saveModelState(const std::vector<torch::Tensor>& w, const std::string& filename);
  std::vector<torch::Tensor> loadModelState(const std::string& filename);
  float getLoss() { return loss; }
  float getErrorRate() { return error_rate; }
};
