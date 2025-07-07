#pragma once

#include "datasetLogic/baseRegDatasetMngr.hpp"
#include "global/globalConstants.hpp"
#include "nets/mnistNet.hpp"
#include "registeredMNIST.hpp"

#include <memory>

/**
 * @brief Class for handling registered MNIST dataset training.
 *
 * Memory layout per sample:
 * [1 uint32_t (original index)][1 int64_t (label)][784 floats (pixels)]
 */
class RegMnistMngr : public BaseRegDatasetMngr<MnistNet> {
private:
  const int IMG_SIZE = 28 * 28; // Size of MNIST image in pixels
  const char *kDataRoot = "./data/mnist";
  const uint32_t DATASET_SIZE = DATASET_SIZE_MNIST;
  torch::optim::SGD optimizer;

  using TrainDataset = decltype(
      RegisteredMNIST(std::declval<RegTrainData&>(), 
                 std::declval<std::unordered_map<size_t, size_t>>())
                 .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                 .map(torch::data::transforms::Stack<>()));

  std::optional<TrainDataset> train_dataset;
  using RegTrainDataLoader = torch::data::StatelessDataLoader<TrainDataset, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> train_loader;

  // Only for the server
  using BuildDataset = decltype (torch::data::datasets::MNIST(kDataRoot));
  std::optional<BuildDataset> build_dataset;

  using TestDataset =
      decltype(torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                   .map(torch::data::transforms::Stack<>()));
  TestDataset test_dataset;

  using TestDataLoader = torch::data::StatelessDataLoader<TestDataset, torch::data::samplers::RandomSampler>;
  std::unique_ptr<TestDataLoader> test_loader;

  void buildLabelToIndicesMap() override;
  void buildRegisteredDataset(const std::vector<size_t> &indices);
  void init();

public:
  RegMnistMngr(int worker_id, TrainInputParams &t_params, MnistNet net);
  ~RegMnistMngr() = default;

  inline void runTraining() override {
    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
      train(epoch, optimizer, *train_loader);
    }
  }

  inline void runInference(const std::vector<torch::Tensor> &w) override {
    runInferenceBase(w, *train_loader);
  }

  inline void runTesting() override { test(*test_loader); }

  void renewDataset(float proportion = 1.0, std::optional<int> seed = std::nullopt) override;
};
