#include "datasetLogic/regMnistMngr.hpp"
#include "datasetLogic/regCIFAR10Mngr.hpp"
#include "datasetLogic/registeredMNIST.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "nets/mnistNet.hpp"
#include "tensorOps.hpp"
#include <vector>

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

  // Needed for subset sampler to partition the dataset
  buildLabelToIndicesMap();

  SubsetSampler train_sampler = get_subset_sampler(
      worker_id, DATASET_SIZE_MNIST, subset_size, t_params.srvr_subset_size, label_to_indices);
  auto &indices = train_sampler.indices();

  // Init reg data structures and pin memory
  initDataInfo(indices, IMG_SIZE);  
  buildRegisteredDataset(indices);
  train_dataset_size = train_dataset->size().value();

  auto loader_temp =
      torch::data::make_data_loader(*train_dataset,
                                    train_sampler,  // Reuse the same sampler
                                    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(loader_temp);

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

  Logger::instance().log("MNIST Registered memory dataset prepared with " +
                         std::to_string(data_info.num_samples) + " samples\n");
}

std::vector<torch::Tensor>
RegMnistMngr::runTraining(int round, const std::vector<torch::Tensor> &w) {
  std::vector<torch::Tensor> w_cuda = updateModelParameters(w);
  size_t param_count = w_cuda.size();

  if (round % 1 == 0) {
    std::cout << "Training model for round " << round
              << " epochs: " << kNumberOfEpochs << "\n";
  }

  Logger::instance().log("MNIST Training model for step " + std::to_string(round) +
                         " epochs: " + std::to_string(kNumberOfEpochs) + "\n");

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, optimizer, *train_loader);
  }

  // if (round % 2 == 0) {
  //   Logger::instance().log("Testing model after training round " +
  //   std::to_string(round) + "\n"); test(model, device, *test_loader,
  //   test_dataset_size);
  // }

  if (device.is_cuda()) {
    return calculateUpdateCuda(w_cuda);
  } else {
    return calculateUpdateCPU(w);
  }
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
}

// Copy data from the original dataset to the registered memory
void RegMnistMngr::buildRegisteredDataset(const std::vector<size_t> &indices) {
  auto plain_mnist = torch::data::datasets::MNIST(kDataRoot);
  std::unordered_map<size_t, size_t> index_map;
  index_map.reserve(indices.size());
  std::vector<size_t> poisoned_labels;

  size_t i = 0;  // Counter for the registered memory
  for (const auto& original_idx : indices) {
    auto example = plain_mnist.get(original_idx);
    int64_t label = example.target.item<int64_t>();

    // Put them at the end of the dataset if they are poisoned
    if (attack_is_targeted_flip && !overwrite_poisoned && label == src_class) {
      poisoned_labels.push_back(original_idx);
      continue;
    }

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
    ++i;
  }

  for (const auto &poisoned_idx : poisoned_labels) {
    auto example = plain_mnist.get(poisoned_idx);
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
    ++i;
  }

  train_dataset = RegisteredMNIST(data_info, index_map)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
}