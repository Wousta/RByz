#include <vector>

#include "datasetLogic/regCIFAR10Mngr.hpp"
#include "datasetLogic/registeredCIFAR10.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"

RegCIFAR10Mngr::RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, ResNet<ResidualBlock> net)
    : BaseRegDatasetMngr<ResNet<ResidualBlock>>(worker_id, t_params, net),
      optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate)),
      build_dataset(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild)
                       .map(ConstantPad(4))
                       .map(RandomHorizontalFlip())
                       .map(RandomCrop({32, 32}))
                       //.map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                       .map(torch::data::transforms::Stack<>())),
      test_dataset(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                       //.map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                       .map(torch::data::transforms::Stack<>())) {

  train_dataset_size = build_dataset.size().value();
  Logger::instance().log("CIFAR10Mngr: train_dataset_size = " +
                         std::to_string(train_dataset_size) + "\n");
  auto build_loader_temp = torch::data::make_data_loader(
      build_dataset,
      torch::data::DataLoaderOptions().batch_size(1));
  build_loader = std::move(build_loader_temp);

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

  if (round % 2 == 0 && round > 1) {
    learning_rate *= learning_rate_decay_factor;
    static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
        .options()).lr(learning_rate);
  }

  if (round % 1 == 0) {
    std::cout << "Training model for round " << round
              << " epochs: " << kNumberOfEpochs << "\n";
  }

  Logger::instance().log("CIFAR10 Training model for step " +
                         std::to_string(round) +
                         " epochs: " + std::to_string(kNumberOfEpochs) + "\n");

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, optimizer, *train_loader);
  }

  if (device.is_cuda()) {
    return calculateUpdateCuda(w_cuda);
  } else {
    return calculateUpdateCPU(w_cuda);
  }
}

void RegCIFAR10Mngr::buildLabelToIndicesMap() {
  label_to_indices.clear();
  size_t i = 0;
  for (auto &example : *build_loader) {
    int64_t label = example.target.item<int64_t>();
    label_to_indices[label].push_back(i++);
  }
}

void RegCIFAR10Mngr::buildRegisteredDataset(const std::vector<size_t> &indices) {
  index_map.reserve(indices.size());
  std::set<size_t> indices_set(indices.begin(), indices.end());

  int og_idx = 0;
  int reg_idx = 0;
  for (auto &example : *build_loader) {
    int64_t label = example.target.item<int64_t>();

    if (indices_set.find(og_idx) != indices_set.end()) {
      *getOriginalIndex(reg_idx) = static_cast<uint32_t>(og_idx);
      *getLabel(reg_idx) = static_cast<int64_t>(label);

      auto normalized_image = (example.data.to(torch::kFloat32)); //- 0.5) / 0.5;
      auto image_tensor = normalized_image.reshape({3, 32, 32}).contiguous();
      std::memcpy(getImage(reg_idx), image_tensor.data_ptr<float>(),
                  data_info.image_size);

      // Map original index to registered index for retrieval
      index_map[og_idx] = reg_idx;
      reg_idx++;
    }

    og_idx++;
  }
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
      .map(RandomHorizontalFlip())
      .map(RandomCrop({32, 32}))
      //.map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
      .map(torch::data::transforms::Stack<>());

  auto loader_temp = torch::data::make_data_loader(
      std::move(*train_dataset),
      train_sampler, // Reuse the same sampler
      torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(loader_temp);
}