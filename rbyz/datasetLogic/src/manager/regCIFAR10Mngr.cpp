#include "manager/regCIFAR10Mngr.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "dataset/registeredCIFAR10.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"

void RegCIFAR10Mngr::init() {
  buildLabelToIndicesMap();

  SubsetSampler train_sampler = get_subset_sampler(worker_id, train_dataset_size, subset_size,
                                                   t_params.srvr_subset_size, label_to_indices);
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

  auto loader_temp =
      torch::data::make_data_loader(std::move(*train_dataset),
                                    train_sampler,  // Reuse the same sampler
                                    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(loader_temp);

  // Build dataset no longer needed except for server for refresh VD samples in RByz
  if (worker_id != 0) {
    build_dataset.reset();
  }
}

RegCIFAR10Mngr::RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, Cifar10Net net)
    : BaseRegDatasetMngr<Cifar10Net>(worker_id, t_params, net),
      optimizer(model->parameters(),
                torch::optim::SGDOptions(learn_rate).momentum(0.9).weight_decay(1e-4)),
      scheduler(optimizer, 82, 0.1),
      build_dataset(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild)),
      test_dataset(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                 {0.2023, 0.1994, 0.2010}))
                       .map(torch::data::transforms::Stack<>())) {
  train_dataset_size = build_dataset->size().value();
  Logger::instance().log("CIFAR10Mngr: train_dataset_size = " + std::to_string(train_dataset_size) +
                         "\n");

  test_dataset_size = test_dataset.size().value();
  auto test_loader_temp = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      test_dataset, torch::data::DataLoaderOptions().batch_size(kTestBatchSize));
  test_loader = std::move(test_loader_temp);

  int label_flip_type = t_params.label_flip_type;
  if (label_flip_type && label_flip_type != RANDOM_FLIP && label_flip_type != TARGETED_FLIP_4) {
    const int target_mappings[3][2] = {
        {5, 3},  // TARGETED_FLIP_1
        {0, 2},  // TARGETED_FLIP_2
        {1, 9}   // TARGETED_FLIP_3
    };

    int setting = label_flip_type - TARGETED_FLIP_1;  // Convert to 0-based index (2->0, 3->1, 4->2)
    src_class = target_mappings[setting][0];
    target_class = target_mappings[setting][1];
    attack_is_targeted_flip = true;
  }

  init();

  Logger::instance().log("CIFAR10 Registered memory dataset prepared with " +
                         std::to_string(data_info.num_samples) + " samples\n");
}

void RegCIFAR10Mngr::buildLabelToIndicesMap() {
  label_to_indices.clear();

  for (size_t i = 0; i < train_dataset_size; i++) {
    int64_t label = build_dataset->get(i).target.item<int64_t>();
    label_to_indices[label].push_back(i);
  }

  // Needs fixed seed so every worker has the same shuffled label_to_indices map, so that subset
  // sampler Distributes the corresponding indices to each worker. This shuffling is not strictly
  // necessary
  std::mt19937 rng(42);
  for (auto &pair : label_to_indices) {
    std::shuffle(pair.second.begin(), pair.second.end(), rng);
  }
}

void RegCIFAR10Mngr::buildRegisteredDataset(const std::vector<size_t> &indices) {
  index_map.reserve(indices.size());
  std::vector<int> labels(10, 0);
  std::vector<size_t> poisoned_labels;

  size_t i = 0;  // Counter for the registered memory
  for (const auto &original_idx : indices) {
    auto example = build_dataset->get(original_idx);
    int64_t label = example.target.item<int64_t>();

    // Put them at the end of the dataset if they are poisoned, server does not poison itself
    if (!overwrite_poisoned && label == src_class && worker_id != 0 &&
        worker_id <= t_params.n_byz_clnts) {
      poisoned_labels.push_back(original_idx);
      continue;
    }
    labels[label]++;

    *getOriginalIndex(i) = static_cast<uint32_t>(original_idx);
    *getLabel(i) = label;

    auto img = (example.data.to(torch::kFloat32));  //- 0.5) / 0.5;
    auto reshaped_img = img.reshape({3, 32, 32}).contiguous();
    std::memcpy(getImage(i), reshaped_img.data_ptr<float>(), data_info.image_size);

    // Map original index to registered index for retrieval
    index_map[original_idx] = i;
    ++i;
  }

  for (const auto &poisoned_idx : poisoned_labels) {
    auto example = build_dataset->get(poisoned_idx);
    int64_t label = example.target.item<int64_t>();
    labels[label]++;

    *getOriginalIndex(i) = static_cast<uint32_t>(poisoned_idx);
    *getLabel(i) = label;

    auto img = (example.data.to(torch::kFloat32));  //- 0.5) / 0.5;
    auto reshaped_img = img.reshape({3, 32, 32}).contiguous();
    std::memcpy(getImage(i), reshaped_img.data_ptr<float>(), data_info.image_size);

    // Map original index to registered index for retrieval
    index_map[poisoned_idx] = i;
    ++i;
  }

  Logger::instance().log("buildRegisteredDataset Labels per class processed: " +
                         std::to_string(labels[0]) + ", " + std::to_string(labels[1]) + ", " +
                         std::to_string(labels[2]) + ", " + std::to_string(labels[3]) + ", " +
                         std::to_string(labels[4]) + ", " + std::to_string(labels[5]) + ", " +
                         std::to_string(labels[6]) + ", " + std::to_string(labels[7]) + ", " +
                         std::to_string(labels[8]) + ", " + std::to_string(labels[9]) + "\n");
}

void RegCIFAR10Mngr::renewDataset(float proportion, std::optional<int> seed) {
  if (worker_id != 0) {
    Logger::instance().log("[renewDataset] Warning: Renewing dataset for worker " +
                           std::to_string(worker_id) +
                           " is not supported, only server can renew dataset.\n");
    return;
  }

  if (proportion <= 0.0f || proportion > 1.0f) {
    throw std::invalid_argument("[renewDataset] Proportion must be between 0 and 1");
  }

  Logger::instance().log("Renewing MNIST dataset with proportion: " + std::to_string(proportion) +
                         "\n");

  if (proportion == 1.0f) {
    for (size_t i = 0; i < data_info.num_samples; i++) {
      auto example = build_dataset->get(renew_idx++);

      if (renew_idx >= build_dataset->size().value()) {
        renew_idx = 0;
      }

      *getLabel(i) = example.target.item<int64_t>();
      auto img = (example.data.to(torch::kFloat32));  //- 0.5) / 0.5;
      auto reshaped_img = img.reshape({3, 32, 32}).contiguous();
      std::memcpy(getImage(i), reshaped_img.data_ptr<float>(), data_info.image_size);
    }

    return;
  }

  std::mt19937 rng;
  if (seed.has_value()) {
    rng = std::mt19937(static_cast<unsigned int>(seed.value()));
  } else {
    rng = std::mt19937(static_cast<unsigned int>(std::time(nullptr)));
  }

  std::vector<size_t> indices(data_info.num_samples);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  uint64_t n_samples = static_cast<uint64_t>(data_info.num_samples * proportion);
  indices.resize(n_samples);

  for (size_t idx : indices) {
    auto example = build_dataset->get(renew_idx++);

    if (renew_idx >= build_dataset->size().value()) {
      renew_idx = 0;
    }

    *getLabel(idx) = example.target.item<int64_t>();
    auto img = (example.data.to(torch::kFloat32));  //- 0.5) / 0.5;
    auto reshaped_img = img.reshape({3, 32, 32}).contiguous();
    std::memcpy(getImage(idx), reshaped_img.data_ptr<float>(), data_info.image_size);
  }
}