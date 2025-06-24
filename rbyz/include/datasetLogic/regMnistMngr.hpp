#pragma once

#include "datasetLogic/baseRegDatasetMngr.hpp"
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

  using RegTrainDataLoader =
      torch::data::StatelessDataLoader<RegisteredMNIST, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> train_loader;
  std::unique_ptr<RegisteredMNIST> train_dataset;

  using DatasetType =
      decltype(torch::data::datasets::MNIST(kDataRoot)
                   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                   .map(torch::data::transforms::Stack<>()));
  DatasetType test_dataset;

  using TestDataLoaderType =
      torch::data::StatelessDataLoader<DatasetType,
                                       torch::data::samplers::RandomSampler>;
  std::unique_ptr<TestDataLoaderType> test_loader;

  void buildLabelToIndicesMap() override;
  void buildRegisteredDataset(const std::vector<size_t> &indices);

public:
  RegMnistMngr(int worker_id, int num_workers, int64_t subset_size,
               MnistNet net);
  ~RegMnistMngr() = default;

  std::vector<torch::Tensor>
  runTraining(int round, const std::vector<torch::Tensor> &w) override;

  inline void runInference(const std::vector<torch::Tensor> &w) override {
    runInferenceBase(w, *train_loader);
  }

  inline void runTesting() override { test(*test_loader); }
};
