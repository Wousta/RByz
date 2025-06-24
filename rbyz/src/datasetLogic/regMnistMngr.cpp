#include "datasetLogic/regMnistMngr.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "nets/mnistNet.hpp"
#include <vector>

RegMnistMngr::RegMnistMngr(int worker_id, int num_workers, int64_t subset_size, MnistNet net)
    : BaseRegDatasetMngr<MnistNet>(worker_id, num_workers, subset_size, net),
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
      worker_id, DATASET_SIZE, subset_size, label_to_indices);
  auto &indices = train_sampler.indices();

  // Init reg data structures and pin memory
  initDataInfo(indices, IMG_SIZE);  
  buildRegisteredDataset(indices);

  // Test getting last samples
  Logger::instance().log("TESTING LAST SAMPLE\n");
  Logger::instance().log("Last sample data: index " +
                         std::to_string(data_info.num_samples - 1) + " og index: " +
                         std::to_string(*getOriginalIndex(data_info.num_samples - 1)) +
                         ", label: " + std::to_string(*getLabel(data_info.num_samples - 1)) +
                         "\n");

  auto loader_temp =
      torch::data::make_data_loader(*train_dataset,
                                    train_sampler,  // Reuse the same sampler
                                    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(loader_temp);

  Logger::instance().log("Registered memory dataset prepared with " +
                         std::to_string(data_info.num_samples) + " samples\n");
}

std::vector<torch::Tensor>
RegMnistMngr::runTraining(int round, const std::vector<torch::Tensor> &w) {
  std::vector<torch::Tensor> w_cuda = updateModelParameters(w);
  size_t param_count = w_cuda.size();

  torch::optim::SGD optimizer(model->parameters(),
                              torch::optim::SGDOptions(GLOBAL_LEARN_RATE));

  if (round % 1 == 0) {
    std::cout << "Training model for round " << round
              << " epochs: " << kNumberOfEpochs << "\n";
  }

  Logger::instance().log("Training model for step " + std::to_string(round) +
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

  size_t i = 0;  // Counter for the registered memory
  for (const auto& original_idx : indices) {
    auto example = plain_mnist.get(original_idx);

    *getOriginalIndex(i) = static_cast<uint32_t>(original_idx);
    *getLabel(i) = static_cast<int64_t>(example.target.item<int64_t>());

    // UINT8CHANGE
    // auto image_tensor = example.data.to(torch::kUInt8); 
    // auto reshaped_image = image_tensor.reshape({1,28, 28}).contiguous();
    // std::memcpy(getImage(i), reshaped_image.data_ptr<uint8_t>(), data_info.image_size);
    auto normalized_image = (example.data.to(torch::kFloat32) - 0.1307) / 0.3081;
    auto reshaped_image = normalized_image.reshape({1, 28, 28}).contiguous();
    std::memcpy(getImage(i), reshaped_image.data_ptr<float>(), data_info.image_size);

    // Map original index to registered index for retrieval
    index_map[original_idx] = i;
    ++i;
  }

  train_dataset = std::make_unique<RegisteredMNIST>(data_info, index_map);
}