#pragma once

#include "datasetLogic/baseRegDatasetMngr.hpp"
#include "registeredMNIST.hpp"

#include <memory>

/**
 * @brief Class for handling registered MNIST dataset training.
 *
 * Memory layout per sample:
 * [1 uint32_t (original index)][1 int64_t (label)][784 floats (pixels)]
 */
class RegMnistMngr : public BaseRegDatasetMngr {
private:
  const int IMG_SIZE = 28 * 28; // Size of MNIST image in pixels

  using RegTrainDataLoader = torch::data::StatelessDataLoader<RegisteredMNIST, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> registered_loader;
  std::unique_ptr<RegisteredMNIST> registered_dataset;
  std::unordered_map<int64_t, std::vector<size_t>> label_to_indices;

  using DatasetType =
      decltype(torch::data::datasets::MNIST(kDataRoot)
                   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                   .map(torch::data::transforms::Stack<>()));
  DatasetType test_dataset;
  
  // Testing is done the same both registered and regular
  using TestDataLoaderType =
      torch::data::StatelessDataLoader<DatasetType,
                                       torch::data::samplers::RandomSampler>;
  std::unique_ptr<TestDataLoaderType> test_loader;

  void buildLabelToIndicesMap() override;
  void buildRegisteredDataset(const std::vector<size_t> &indices);

public:
  RegMnistMngr(int worker_id, int num_workers, int64_t subset_size,
               std::unique_ptr<NNet> net);
  ~RegMnistMngr() override;

  std::vector<torch::Tensor> runTraining(int round, const std::vector<torch::Tensor> &w) override;

  inline void runInference(const std::vector<torch::Tensor> &w) override {
    runInferenceBase(w, *registered_loader);
  }

  inline void runTesting() override {
    test(*test_loader);
  }
};
