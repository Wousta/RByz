#pragma once

#include "datasetLogic/baseRegDatasetMngr.hpp"
#include "global/globalConstants.hpp"
#include "nets/cifar10Net.hpp"
#include "transform.hpp"
#include "nets/resnet.hpp"
#include "registeredCIFAR10.hpp"

using resnet::ResNet;
using resnet::ResidualBlock;
using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

class RegCIFAR10Mngr : public BaseRegDatasetMngr<ResNet<ResidualBlock>> {
private:
  // Size of CIFAR-10 image in pixels (3 channels)
  const int IMG_SIZE = 32 * 32 * 3;
  const std::string kDataRoot = "./data/cifar10";
  const double learning_rate_decay_factor = 1.0 / 3.0;
  uint32_t train_dataset_size = 0;
  std::unordered_map<size_t, size_t> index_map;
  torch::optim::Adam optimizer;

  using TrainDataset = decltype(
      RegCIFAR10(std::declval<RegTrainData&>(), 
                 std::declval<std::unordered_map<size_t, size_t>>())
      .map(ConstantPad(4))
      .map(RandomHorizontalFlip())
      .map(RandomCrop({32, 32}))
      //.map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
      .map(torch::data::transforms::Stack<>()));
  
  std::optional<TrainDataset> train_dataset;

  using RegTrainDataLoader =
      torch::data::StatelessDataLoader<TrainDataset, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> train_loader;

  using DatasetType =
      decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                //.map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                .map(torch::data::transforms::Stack<>()));
  DatasetType test_dataset;

  using BuildDataset = decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild)
                .map(ConstantPad(4))
                .map(RandomHorizontalFlip())
                .map(RandomCrop({32, 32}))
                //.map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                .map(torch::data::transforms::Stack<>()));
  BuildDataset build_dataset;

  using BuildDataLoaderType =
      torch::data::StatelessDataLoader<BuildDataset,
                                       torch::data::samplers::RandomSampler>;
  std::unique_ptr<BuildDataLoaderType> build_loader;

  using TestDataLoaderType =
      torch::data::StatelessDataLoader<DatasetType,
                                       torch::data::samplers::RandomSampler>;
  std::unique_ptr<TestDataLoaderType> test_loader;

  void buildLabelToIndicesMap() override;
  void buildRegisteredDataset(const std::vector<size_t> &indices);
  void init();

public:
  RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, ResNet<ResidualBlock> net);
  ~RegCIFAR10Mngr() = default;

  std::vector<torch::Tensor>
  runTraining(int round, const std::vector<torch::Tensor> &w) override;

  inline void runInference(const std::vector<torch::Tensor> &w) override {
    runInferenceBase(w, *train_loader);
  }

  inline void runTesting() override { test(*test_loader); }
};
