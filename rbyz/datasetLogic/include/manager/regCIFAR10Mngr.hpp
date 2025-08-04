#pragma once

#include "dataset/registeredCIFAR10.hpp"
#include "manager/baseRegDatasetMngr.hpp"
#include "net/cifar10Net.hpp"
#include "net/resnet.hpp"
#include "transform.hpp"

using resnet::ResidualBlock;
using resnet::ResNet;
using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

// class RegCIFAR10Mngr : public BaseRegDatasetMngr<ResNet<ResidualBlock>> {
class RegCIFAR10Mngr : public BaseRegDatasetMngr<Cifar10Net> {
 private:
  // Size of CIFAR-10 image in pixels (3 channels)
  const int IMG_SIZE = 32 * 32 * 3;
  const std::string kDataRoot = "./data/cifar10";
  const double learning_rate_decay_factor = 1.0 / 3.0;
  std::unordered_map<size_t, size_t> index_map;
  // torch::optim::Adam optimizer;
  torch::optim::SGD optimizer;
  torch::optim::StepLR scheduler;

  using TrainDataset = decltype(RegCIFAR10(std::declval<RegTrainData &>(),
                                           std::declval<std::unordered_map<size_t, size_t>>())
                                    .map(ConstantPad(4))
                                    .map(RandomCrop({32, 32}))
                                    .map(RandomHorizontalFlip())
                                    .map(torch::data::transforms::Normalize<>(
                                        {0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>()));
  std::optional<TrainDataset> train_dataset;

  using RegTrainDataLoader = torch::data::StatelessDataLoader<TrainDataset, SubsetSampler>;
  std::unique_ptr<RegTrainDataLoader> train_loader;

  using TestDataset = decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                                   .map(torch::data::transforms::Normalize<>(
                                       {0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                   .map(torch::data::transforms::Stack<>()));
  TestDataset test_dataset;

  using TestDataLoader =
      torch::data::StatelessDataLoader<TestDataset, torch::data::samplers::SequentialSampler>;
  std::unique_ptr<TestDataLoader> test_loader;

  using BuildDataset = decltype(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild));
  std::optional<BuildDataset> build_dataset;

  void buildLabelToIndicesMap() override;
  void buildRegisteredDataset(const std::vector<size_t> &indices);
  void init();

 public:
  RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, Cifar10Net net);
  // RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, ResNet<ResidualBlock> net);
  ~RegCIFAR10Mngr() = default;

  inline void runTraining() override {
    // if (round % 5 == 0 && round > 1) {
    //   learning_rate *= learning_rate_decay_factor;
    //   static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
    //       .options()).lr(learn_rate);
    // }
    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
      train(epoch, optimizer, *train_loader);
      scheduler.step();
    }
  }

  inline void runInference() override { runInferenceBase(*train_loader); }
  inline void runTesting() override { test(*test_loader); }
  void renewDataset(float proportion = 1.0, std::optional<int> seed = std::nullopt) override;
};
