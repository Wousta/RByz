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
// class RegCIFAR10Mngr : public BaseRegDatasetMngr<Cifar10Net> {
private:
  // Size of CIFAR-10 image in pixels (3 channels)
  const int IMG_SIZE = 32 * 32 * 3;
  const std::string kDataRoot = "./data/cifar10";
  const double learning_rate_decay_factor = 1.0 / 3.0;
  std::unordered_map<size_t, size_t> index_map;
  //torch::optim::Adam optimizer;
  torch::optim::SGD optimizer; 
  torch::optim::StepLR scheduler;

  using TrainDataset = decltype(
      RegCIFAR10(std::declval<RegTrainData&>(), 
                 std::declval<std::unordered_map<size_t, size_t>>())
      .map(ConstantPad(4))
      .map(RandomCrop({32, 32}))
      .map(RandomHorizontalFlip())
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, 
                                                         {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>()));
  std::optional<TrainDataset> train_dataset;

  using RegTrainDataLoader = torch::data::StatelessDataLoader<TrainDataset, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> train_loader;

  using TestDataset =
      decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, 
                                                                  {0.2023, 0.1994, 0.2010}))
                .map(torch::data::transforms::Stack<>()));
  TestDataset test_dataset;

  using TestDataLoader =
      torch::data::StatelessDataLoader<TestDataset,
                                       torch::data::samplers::SequentialSampler>;
  std::unique_ptr<TestDataLoader> test_loader;

  using BuildDataset = decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild));
  std::optional<BuildDataset> build_dataset;

  void buildLabelToIndicesMap() override;
  void buildRegisteredDataset(const std::vector<size_t> &indices);
  void init();

public:
  // RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, Cifar10Net net);
  RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, ResNet<ResidualBlock> net);
  ~RegCIFAR10Mngr() = default;

  std::vector<torch::Tensor>
  runTraining(int round, const std::vector<torch::Tensor> &w) override;

  inline void runInference(const std::vector<torch::Tensor> &w) override {
    runInferenceBase(w, *train_loader);
  }

  inline void runTesting() override { test(*test_loader); }

  void renewDataset(float proportion = 1.0, std::optional<int> seed = std::nullopt) override;
};
