#pragma once

#include "datasetLogic/baseRegDatasetMngr.hpp"
#include "global/globalConstants.hpp"
#include "nets/cifar10Net.hpp"
#include "registeredCIFAR10.hpp"

class RegCIFAR10Mngr : public BaseRegDatasetMngr<Cifar10Net> {
private:
  // Size of CIFAR-10 image in pixels (3 channels)
  const int IMG_SIZE = 32 * 32 * 3;
  const uint32_t DATASET_SIZE = DATASET_SIZE_CF10;
  const std::string kDataRoot = "./data/cifar10";

  using RegTrainDataLoader =
      torch::data::StatelessDataLoader<RegCIFAR10, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> train_loader;
  std::unique_ptr<RegCIFAR10> train_dataset;

  using DatasetType =
      decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                .map(torch::data::transforms::Stack<>()));
  DatasetType test_dataset;

  using BuildDataset = decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild)
                .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                .map(torch::data::transforms::Stack<>()));
  using BuildDataLoaderType =
      torch::data::StatelessDataLoader<BuildDataset,
                                       torch::data::samplers::RandomSampler>;

  using TestDataLoaderType =
      torch::data::StatelessDataLoader<DatasetType,
                                       torch::data::samplers::RandomSampler>;
  std::unique_ptr<TestDataLoaderType> test_loader;

  void buildLabelToIndicesMap() override;
  void buildRegisteredDataset(const std::vector<size_t> &indices);

public:
  RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, Cifar10Net net);
  ~RegCIFAR10Mngr() = default;

  std::vector<torch::Tensor>
  runTraining(int round, const std::vector<torch::Tensor> &w) override;

  inline void runInference(const std::vector<torch::Tensor> &w) override {
    runInferenceBase(w, *train_loader);
  }

  inline void runTesting() override { test(*test_loader); }
};
