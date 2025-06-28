#include <vector>

#include "datasetLogic/regCIFAR10Mngr.hpp"
#include "datasetLogic/registeredCIFAR10.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"

RegCIFAR10Mngr::RegCIFAR10Mngr(int worker_id, TrainInputParams &t_params, ResNet<ResidualBlock> net)
    : BaseRegDatasetMngr<ResNet<ResidualBlock>>(worker_id, t_params, net),
      test_dataset(RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>(
                           {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                       .map(torch::data::transforms::Stack<>())) {

  test_dataset_size = test_dataset.size().value();
  auto test_loader_temp = torch::data::make_data_loader(
      test_dataset,
      torch::data::DataLoaderOptions().batch_size(kTestBatchSize));
  test_loader = std::move(test_loader_temp);

  // Needed for subset sampler to partition the dataset
  buildLabelToIndicesMap();

  SubsetSampler train_sampler = get_subset_sampler(
      worker_id, DATASET_SIZE, subset_size, t_params.srvr_subset_size, label_to_indices);
  auto &indices = train_sampler.indices();

  // Init reg data structures and pin memory
  initDataInfo(indices, IMG_SIZE);
  buildRegisteredDataset(indices);

  auto loader_temp = torch::data::make_data_loader(
      *train_dataset,
      train_sampler, // Reuse the same sampler
      torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(loader_temp);

  Logger::instance().log("CIFAR10 Registered memory dataset prepared with " +
                         std::to_string(data_info.num_samples) + " samples\n");
}

std::vector<torch::Tensor>
RegCIFAR10Mngr::runTraining(int round, const std::vector<torch::Tensor> &w) {
  std::vector<torch::Tensor> w_cuda = updateModelParameters(w);
  size_t param_count = w_cuda.size();

  auto optimizer = torch::optim::Adam(model->parameters(), 
                                    torch::optim::AdamOptions(0.001));

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
  RegCIFAR10 dataset(kDataRoot, RegCIFAR10::Mode::kBuild);

  for (size_t i = 0; i < dataset.size().value(); ++i) {
    auto example = dataset.get(i);
    int64_t label = example.target.item<int64_t>();
    label_to_indices[label].push_back(i);
  }
}

void RegCIFAR10Mngr::buildRegisteredDataset(
    const std::vector<size_t> &indices) {
  auto plain_cifar10 = RegCIFAR10(kDataRoot, RegCIFAR10::Mode::kBuild);
  std::unordered_map<size_t, size_t> index_map;
  index_map.reserve(indices.size());

  size_t i = 0; // Counter for the registered memory
  for (const auto &original_idx : indices) {
    auto example = plain_cifar10.get(original_idx);

    *getOriginalIndex(i) = static_cast<uint32_t>(original_idx);
    *getLabel(i) = static_cast<int64_t>(example.target.item<int64_t>());

    // Copy image data
    auto normalized_image = (example.data.to(torch::kFloat32) - 0.5) / 0.5;
    auto image_tensor = normalized_image.reshape({3, 32, 32}).contiguous();
    std::memcpy(getImage(i), image_tensor.data_ptr<float>(),
                data_info.image_size);

    // Map original index to registered index for retrieval
    index_map[original_idx] = i;
    ++i;
  }

  train_dataset = std::make_unique<RegCIFAR10>(data_info, index_map);
}