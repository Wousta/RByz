#include "datasetLogic/regMnistMngr.hpp"
#include "datasetLogic/regCIFAR10Mngr.hpp"
#include "datasetLogic/registeredMNIST.hpp"
#include "datasetLogic/subsetSampler.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "nets/mnistNet.hpp"
#include "tensorOps.hpp"
#include <string>
#include <vector>

void RegMnistMngr::init() {
  // To refresh VD samples in RByz
  build_dataset = torch::data::datasets::MNIST(kDataRoot);
  train_dataset_size = build_dataset->size().value();

  // Needed for subset sampler to partition the dataset
  buildLabelToIndicesMap();

  SubsetSampler train_sampler = get_subset_sampler(
      worker_id, DATASET_SIZE_MNIST, subset_size, t_params.srvr_subset_size, label_to_indices);
  auto &indices = train_sampler.indices();

  // Init reg data structures and pin memory
  initDataInfo(indices, IMG_SIZE);  
  buildRegisteredDataset(indices);

  auto loader_temp =
      torch::data::make_data_loader(*train_dataset,
                                    train_sampler,  // Reuse the same sampler
                                    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(loader_temp);

  if (worker_id != 0) {
    // Build dataset no longer needed except for server for refresh VD samples in RByz
    build_dataset.reset();
  }
    
}

RegMnistMngr::RegMnistMngr(int worker_id, TrainInputParams &t_params, MnistNet net)
    : BaseRegDatasetMngr<MnistNet>(worker_id, t_params, net),
      optimizer(model->parameters(), torch::optim::SGDOptions(learn_rate)),
      test_dataset(
          torch::data::datasets::MNIST(
              kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
              .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
              .map(torch::data::transforms::Stack<>())) {

  test_dataset_size = test_dataset.size().value();
  auto test_loader_temp = torch::data::make_data_loader(
      test_dataset,
      torch::data::DataLoaderOptions().batch_size(kTestBatchSize));
  test_loader = std::move(test_loader_temp);

  Logger::instance().log("MNISTMngr: label_flip_type = " + std::to_string(t_params.label_flip_type) +
                          ", overwrite_poisoned = " + std::to_string(t_params.overwrite_poisoned) +
                          ", worker id = " + std::to_string(worker_id) + "\n");

  int label_flip_type = t_params.label_flip_type;
  if (label_flip_type && label_flip_type != RANDOM_FLIP) {
    const int target_mappings[3][2] = {
      {8, 0}, // TARGETED_FLIP_1
      {1, 5}, // TARGETED_FLIP_2  
      {4, 9}  // TARGETED_FLIP_3
    };
    int setting = label_flip_type - TARGETED_FLIP_1; // Convert to 0-based index (2->0, 3->1, 4->2)
    src_class = target_mappings[setting][0];
    target_class = target_mappings[setting][1];
    attack_is_targeted_flip = true;
  }

  init();

  Logger::instance().log("MNIST Registered memory dataset prepared with " +
                         std::to_string(data_info.num_samples) + " samples\n");
}

void RegMnistMngr::buildLabelToIndicesMap() {
  label_to_indices.clear();
  auto dataset = torch::data::datasets::MNIST(kDataRoot)
                     .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                     .map(torch::data::transforms::Stack<>());

  size_t index = 0;
  // Create a DataLoader to iterate over the dataset
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          dataset, /*batch size*/ 1);

  for (const auto &example : *data_loader) {
    int64_t label = example.target.item<int64_t>();
    label_to_indices[label].push_back(index);
    ++index;
  }

  // Needs fixed seed so every worker has the same shuffled label_to_indices map, so that subset sampler 
  // Distributes the corresponding indices to each worker, this shuffling is not strictly necessary
  std::mt19937 rng(42);
  for (auto &pair : label_to_indices) {
    std::shuffle(pair.second.begin(), pair.second.end(), rng);
  }
}

// Copy data from the original dataset to the registered memory
void RegMnistMngr::buildRegisteredDataset(const std::vector<size_t> &indices) {
  std::unordered_map<size_t, size_t> index_map;
  index_map.reserve(indices.size());
  std::vector<size_t> poisoned_labels;

  size_t i = 0;  // Counter for the registered memory
  for (const auto& original_idx : indices) {
    auto example = build_dataset->get(original_idx);
    int64_t label = example.target.item<int64_t>();

    // // Put them at the end of the dataset if they are poisoned
    // if (!overwrite_poisoned && label == src_class && worker_id != 0 && worker_id <= t_params.n_byz_clnts) {
    //   poisoned_labels.push_back(original_idx);
    //   continue;
    // }

    *getOriginalIndex(i) = static_cast<uint32_t>(original_idx);
    *getLabel(i) = label;

    // UINT8CHANGE
    // auto image_tensor = example.data.to(torch::kUInt8); 
    // auto reshaped_image = image_tensor.reshape({1,28, 28}).contiguous();
    // std::memcpy(getImage(i), reshaped_image.data_ptr<uint8_t>(), data_info.image_size);
    auto normalized_image = example.data.to(torch::kFloat32);
    auto reshaped_image = normalized_image.reshape({1, 28, 28}).contiguous();
    std::memcpy(getImage(i), reshaped_image.data_ptr<float>(), data_info.image_size);

    // Map original index to registered index for retrieval
    index_map[original_idx] = i;
    i++;
  }

  for (const auto &poisoned_idx : poisoned_labels) {
    auto example = build_dataset->get(poisoned_idx);
    int64_t label = example.target.item<int64_t>();

    *getOriginalIndex(i) = static_cast<uint32_t>(poisoned_idx);
    *getLabel(i) = label;

    // UINT8CHANGE
    // auto image_tensor = example.data.to(torch::kUInt8); 
    // auto reshaped_image = image_tensor.reshape({1,28, 28}).contiguous();
    // std::memcpy(getImage(i), reshaped_image.data_ptr<uint8_t>(), data_info.image_size);
    auto normalized_image = example.data.to(torch::kFloat32);
    auto reshaped_image = normalized_image.reshape({1, 28, 28}).contiguous();
    std::memcpy(getImage(i), reshaped_image.data_ptr<float>(), data_info.image_size);

    index_map[poisoned_idx] = i;
    i++;
  }

  train_dataset = RegisteredMNIST(data_info, index_map)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
}

void RegMnistMngr::renewDataset(float proportion, std::optional<int> seed) {
  if (worker_id != 0) {
    Logger::instance().log("[renewDataset] Warning: Renewing dataset for worker " + std::to_string(worker_id) +
                           " is not supported, only server can renew dataset.\n");
    return;
  }

  if (proportion <= 0.0f || proportion > 1.0f) {
    throw std::invalid_argument("[renewDataset] Proportion must be between 0 and 1");
  }

  Logger::instance().log("Renewing MNIST dataset with proportion: " +
                        std::to_string(proportion) + "\n");

  if (proportion == 1.0f) {
    for (size_t i = 0; i < data_info.num_samples; i++) {
      auto example = build_dataset->get(renew_idx++);

      if (renew_idx >= build_dataset->size().value()) {
        renew_idx = 0;
      }

      *getLabel(i) = example.target.item<int64_t>();
      auto image = example.data.to(torch::kFloat32);
      auto reshaped_image = image.reshape({1, 28, 28}).contiguous();
      std::memcpy(getImage(i), reshaped_image.data_ptr<float>(), data_info.image_size);
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
    auto image = example.data.to(torch::kFloat32);
    auto reshaped_image = image.reshape({1, 28, 28}).contiguous();
    std::memcpy(getImage(idx), reshaped_image.data_ptr<float>(), data_info.image_size);
  }
}