#include <cstdint>
#include <vector>

#include "datasetLogic/regCIFAR10Mngr.hpp"
#include "datasetLogic/registeredCIFAR10.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"

RegCIFAR10Mngr::RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, ResNet<ResidualBlock> net)
    : BaseRegDatasetMngr<ResNet<ResidualBlock>>(worker_id, t_params, net),
      //optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate)),
      optimizer(model->parameters(), torch::optim::SGDOptions(learn_rate)
          .momentum(0.9) 
          .weight_decay(1e-4)), 
      scheduler(optimizer, 82, 0.1),
      build_dataset(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild)),
      test_dataset(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, 
                                                                          {0.2023, 0.1994, 0.2010}))
                       .map(torch::data::transforms::Stack<>())) {

  train_dataset_size = build_dataset->size().value();
  Logger::instance().log("CIFAR10Mngr: train_dataset_size = " +
                         std::to_string(train_dataset_size) + "\n");

  test_dataset_size = test_dataset.size().value();
  auto test_loader_temp = torch::data::make_data_loader(
      test_dataset,
      torch::data::DataLoaderOptions().batch_size(kTestBatchSize));
  test_loader = std::move(test_loader_temp);

  init();

  Logger::instance().log("CIFAR10 Registered memory dataset prepared with " +
                         std::to_string(data_info.num_samples) + " samples\n");
}

std::vector<torch::Tensor>
RegCIFAR10Mngr::runTraining(int round, const std::vector<torch::Tensor> &w) {
  std::vector<torch::Tensor> w_cuda = updateModelParameters(w);
  size_t param_count = w_cuda.size();

  // if (round % 5 == 0 && round > 1) {
  //   learning_rate *= learning_rate_decay_factor;
  //   static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
  //       .options()).lr(learn_rate);
  // }

  if (round % 1 == 0) {
    std::cout << "Training model for round " << round
              << " epochs: " << kNumberOfEpochs << "\n";
  }

  Logger::instance().log("CIFAR10 Training model for step " +
                         std::to_string(round) +
                         " epochs: " + std::to_string(kNumberOfEpochs) + "\n");

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, optimizer, *train_loader);
    scheduler.step();
  }

  if (device.is_cuda()) {
    return calculateUpdateCuda(w_cuda);
  } else {
    return calculateUpdateCPU(w_cuda);
  }
}

void RegCIFAR10Mngr::buildLabelToIndicesMap() {
  label_to_indices.clear();

  for (size_t i = 0; i < train_dataset_size; i++) {
    int64_t label = build_dataset->get(i).target.item<int64_t>();
    label_to_indices[label].push_back(i);
  }

  std::mt19937 rng(42); // Fixed seed for reproducible shuffling
  for (auto &pair : label_to_indices) {
    std::shuffle(pair.second.begin(), pair.second.end(), rng);
  }
}

void RegCIFAR10Mngr::buildRegisteredDataset(const std::vector<size_t> &indices) {
  index_map.reserve(indices.size());
  std::vector<int> labels(10, 0);

  size_t i = 0;  // Counter for the registered memory
  for (const auto& original_idx : indices) {
    auto example = build_dataset->get(original_idx);
    int64_t label = example.target.item<int64_t>();
    labels[label]++;

    *getOriginalIndex(i) = static_cast<uint32_t>(original_idx);
    *getLabel(i) = label;

      auto img = (example.data.to(torch::kFloat32)); //- 0.5) / 0.5;
      auto reshaped_img = img.reshape({3, 32, 32}).contiguous();
      std::memcpy(getImage(i), reshaped_img.data_ptr<float>(),
                  data_info.image_size);

    // Map original index to registered index for retrieval
    index_map[original_idx] = i;
    ++i;
  }

  Logger::instance().log("buildRegisteredDataset Labels per class processed: " +
                        std::to_string(labels[0]) + ", " +
                        std::to_string(labels[1]) + ", " +
                        std::to_string(labels[2]) + ", " +
                        std::to_string(labels[3]) + ", " +
                        std::to_string(labels[4]) + ", " +
                        std::to_string(labels[5]) + ", " +
                        std::to_string(labels[6]) + ", " +
                        std::to_string(labels[7]) + ", " +
                        std::to_string(labels[8]) + ", " +
                        std::to_string(labels[9]) + "\n");
}

void RegCIFAR10Mngr::init() {
  buildLabelToIndicesMap();

  SubsetSampler train_sampler = get_subset_sampler(
      worker_id, train_dataset_size, subset_size, t_params.srvr_subset_size, label_to_indices);
  auto &indices = train_sampler.indices();

  // Init reg data structures and pin memory
  initDataInfo(indices, IMG_SIZE);
  buildRegisteredDataset(indices);

  train_dataset = RegCIFAR10(data_info, index_map)
      .map(ConstantPad(4))
      .map(RandomCrop({32, 32}))
      .map(RandomHorizontalFlip())
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, 
                                                        {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());

  auto loader_temp = torch::data::make_data_loader(
      std::move(*train_dataset),
      train_sampler, // Reuse the same sampler
      torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(loader_temp);

  // Build dataset no longer needed, destroy
  build_dataset.reset();
}