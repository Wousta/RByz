// Code from:
// https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/popular/blitz/training_a_classifier/src/cifar10.cpp

// Copyright 2020-present pytorch-cpp Authors
#include "datasetLogic/registeredCIFAR10.hpp"

namespace {
// CIFAR10 dataset description can be found at
// https://www.cs.toronto.edu/~kriz/cifar.html.
constexpr uint32_t kTrainSize = 50000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kSizePerBatch = 10000;
constexpr uint32_t kImageRows = 32;
constexpr uint32_t kImageColumns = 32;
constexpr uint32_t kBytesPerRow = 3073;
constexpr uint32_t kBytesPerChannelPerRow = 1024;
constexpr uint32_t kBytesPerBatchFile = kBytesPerRow * kSizePerBatch;

const std::vector<std::string> kTrainDataBatchFiles = {
    "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
    "data_batch_4.bin", "data_batch_5.bin",
};

const std::vector<std::string> kTestDataBatchFiles = {"test_batch.bin"};

// Source:
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp.
std::string join_paths(std::string head, const std::string &tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}
// Partially based on
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp.
std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &root,
                                                  bool train) {
  const auto &files = train ? kTrainDataBatchFiles : kTestDataBatchFiles;
  const auto num_samples = train ? kTrainSize : kTestSize;

  std::vector<char> data_buffer;
  data_buffer.reserve(files.size() * kBytesPerBatchFile);

  for (const auto &file : files) {
    const auto path = join_paths(root, file);
    std::ifstream data(path, std::ios::binary);
    TORCH_CHECK(data, "Error opening data file at", path);

    data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data),
                       {});
  }

  TORCH_CHECK(data_buffer.size() == files.size() * kBytesPerBatchFile,
              "Unexpected file sizes");

  auto targets = torch::empty(num_samples, torch::kByte);
  auto images =
      torch::empty({num_samples, 3, kImageRows, kImageColumns}, torch::kByte);

  for (uint32_t i = 0; i != num_samples; ++i) {
    // The first byte of each row is the target class index.
    uint32_t start_index = i * kBytesPerRow;
    targets[i] = data_buffer[start_index];

    // The next bytes correspond to the rgb channel values in the following
    // order: red (32 *32 = 1024 bytes) | green (1024 bytes) | blue (1024 bytes)
    uint32_t image_start = start_index + 1;
    uint32_t image_end = image_start + 3 * kBytesPerChannelPerRow;
    std::copy(data_buffer.begin() + image_start,
              data_buffer.begin() + image_end,
              reinterpret_cast<char *>(images[i].data_ptr()));
  }

  return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}
} // namespace

RegCIFAR10::RegCIFAR10(const std::string &root, Mode mode)
    : mode_(mode),
      // options(torch::TensorOptions().dtype(torch::kUInt8)) { // UINT8CHANGE
      options(torch::TensorOptions().dtype(torch::kFloat32)) {

  if (mode == Mode::kTest || mode == Mode::kBuild) {
    auto data = read_data(root, mode == Mode::kBuild);

    images_ = std::move(data.first);
    targets_ = std::move(data.second);
  }
}

RegCIFAR10::RegCIFAR10(RegTrainData &data_info, std::unordered_map<size_t, size_t> index_map)
    : data_info(&data_info), 
      index_map(std::move(index_map)),
      options(torch::TensorOptions().dtype(torch::kFloat32)) {

  num_samples = data_info.num_samples;
  index_size = data_info.index_size;
  label_size = data_info.label_size;
  image_size = data_info.image_size;
  sample_size = data_info.get_sample_size();
}

torch::data::Example<> RegCIFAR10::get(size_t original_index) {
  if (mode_ == Mode::kTest || mode_ == Mode::kBuild) {
    return {images_[original_index], targets_[original_index]};
  }

  // For training, we need to use the index map to find the correct index
  size_t index = index_map.at(original_index);

  if (index >= num_samples) {
    throw std::out_of_range("Index out of range in RegCIFAR10::get()");
  }

  void *sample = static_cast<char *>(data_info->reg_data) + (index * sample_size);
  //uint8_t* img_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(sample) + index_size + label_size);
  float *img_ptr = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(sample) + index_size + label_size);

  torch::Tensor image = torch::from_blob(
        img_ptr, 
        {3, kImageRows, kImageColumns},  // [channels, height, width]
        options
  ).clone();

  //auto data_normalized = (image.to(torch::kFloat32) - 0.1307f) / 0.3081f; // UINT8CHANGE

  int64_t label = *reinterpret_cast<int64_t*>(reinterpret_cast<uint8_t*>(sample) + index_size);
  if (label < 0 || label >= 10) {
      throw std::runtime_error("Invalid label in RegisteredMNIST::get(): " + std::to_string(label));
  }
  torch::Tensor target = torch::tensor(label, torch::kInt64);

  //return {data_normalized, target}; // UINT8CHANGE
  return {image, target};
}

torch::optional<size_t> RegCIFAR10::size() const { 
  if (mode_ == Mode::kTrain) {
    return num_samples;
  }
  return images_.size(0); 
}

bool RegCIFAR10::is_train() const noexcept { return mode_ == Mode::kTrain; }

const torch::Tensor &RegCIFAR10::images() const { 
  if (mode_ == Mode::kTrain) {
    throw std::runtime_error("[CIFAR10] Cannot access images_ in training mode");
  }
  return images_; 
}

const torch::Tensor &RegCIFAR10::targets() const { 
  if (mode_ == Mode::kTrain) {
    throw std::runtime_error("[CIFAR10] Cannot access targets_ in training mode");
  }
  return targets_; 
}