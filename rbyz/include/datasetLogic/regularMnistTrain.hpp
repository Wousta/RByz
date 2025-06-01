#pragma once

#include "baseMnistTrain.hpp"
#include <memory>

class RegularMnistTrain : public BaseMnistTrain {
private:
  using DatasetType = decltype(
    torch::data::datasets::MNIST(kDataRoot)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>())
  );

  using TrainDataLoaderType = torch::data::StatelessDataLoader<DatasetType, SubsetSampler>;
  
  DatasetType train_dataset;
  std::unique_ptr<TrainDataLoaderType> train_loader;

  template <typename DataLoader>
  void train(
      size_t epoch,
      Net& model,
      torch::Device device,
      DataLoader& data_loader,
      torch::optim::Optimizer& optimizer,
      size_t dataset_size);

public:
  RegularMnistTrain(int worker_id, int num_workers, int64_t subset_size);
  ~RegularMnistTrain() = default;

  std::vector<torch::Tensor> runMnistTrain(int round, const std::vector<torch::Tensor>& w) override;
  void runInference(const std::vector<torch::Tensor>& w) override;
};