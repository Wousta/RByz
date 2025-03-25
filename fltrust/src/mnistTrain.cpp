#include "../include/mnistTrain.hpp"
#include "../include/logger.hpp"
#include "../include/subsetSampler.hpp"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int counter = 1;

// Where to find the MNIST dataset.
const int64_t kTrainBatchSize = 1;
const int64_t kTestBatchSize = 1;
const int64_t kNumberOfEpochs = 1;
const int64_t kLogInterval = 1;
const double learnRate = 0.001;
const int64_t subset_size = 150;

// auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
//   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//   .map(torch::data::transforms::Stack<>());
// const size_t train_dataset_size = train_dataset.size().value();

// auto test_dataset = torch::data::datasets::MNIST(
//   kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
//   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//   .map(torch::data::transforms::Stack<>());
// const size_t test_dataset_size = test_dataset.size().value();

SubsetSampler get_subset_sampler(size_t dataset_size, int64_t subset_size) {
  auto indices_tensor = torch::randperm(dataset_size);
  auto subset_tensor = indices_tensor.slice(0, 0, subset_size);

  std::vector<size_t> subset_indices(subset_size);
  for (int64_t i = 0; i < subset_size; ++i) {
    subset_indices[i] = static_cast<size_t>(subset_tensor[i].item<int64_t>());
  }

  return SubsetSampler(subset_indices);
}

// SubsetSampler train_sampler = get_subset_sampler(train_dataset_size, subset_size);

// auto train_loader = torch::data::make_data_loader(
//   std::move(train_dataset),
//   train_sampler,
//   torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));

// auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);


MnistTrain::MnistTrain() 
  : device(init_device()),
  train_dataset(torch::data::datasets::MNIST(kDataRoot)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>())),
  test_dataset(torch::data::datasets::MNIST(
    kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>()))
  {

  if (torch::cuda::is_available()) {
    Logger::instance().log("CUDA available! Training on GPU.\n");
    device_type = torch::kCUDA;
  } else {
    Logger::instance().log("Training on CPU.\n");
    device_type = torch::kCPU;
  }

  train_dataset_size = train_dataset.size().value();
  test_dataset_size = test_dataset.size().value();
  model.to(device);
}

MnistTrain::~MnistTrain() {
  // Do nothing
}

torch::Device MnistTrain::init_device() {
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    device_type = torch::kCUDA;
  } else {
    device_type = torch::kCPU;
  }
  return torch::Device(device_type);
}

std::vector<torch::Tensor> MnistTrain::runMnistTrainDummy(std::vector<torch::Tensor>& w) {
  std::cout << "Running dummy MNIST training\n";
  
  for(size_t i = 0; i < w.size(); i++) {
    w[i] = w[i] + 1;  // element-wise addition of 1
  }
  
  return w;
}

template <typename DataLoader>
void train(
    size_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
  Logger::instance().log("Testing done\n");
}

std::vector<torch::Tensor> MnistTrain::runMnistTrain(const std::vector<torch::Tensor>& w) {
  torch::manual_seed(1);
  
  // Update model parameters with input weights if sizes match
  auto params = model.parameters();
  if (w.size() == params.size()) {
    std::cout << "\nUpdating model parameters with input weights." << std::endl;
    for (size_t i = 0; i < params.size(); ++i) {
      // Copy the input weight tensor to the corresponding model parameter
      params[i].data().copy_(w[i]);
    }
  } else if (!w.empty()) {
    {
      std::ostringstream oss;
      oss << "Warning: Input weight size (" << w.size() << ") does not match model parameter size (" ;
      oss << params.size() << "). Using default initialization." << std::endl;
      Logger::instance().log(oss.str());
    }
  }

  SubsetSampler train_sampler = get_subset_sampler(train_dataset_size, subset_size);

  auto train_loader = torch::data::make_data_loader(
    train_dataset,
    train_sampler,
    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));

  auto test_loader = torch::data::make_data_loader(test_dataset, kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(learnRate).momentum(1));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, subset_size);
    //test(model, device, *test_loader, test_dataset_size);
  }
  
  // Extract model weights after training
  std::vector<torch::Tensor> model_weights;
  for (const auto& param : model.parameters()) {
    // Clone the parameter to ensure we're not just storing references
    model_weights.push_back(param.clone().detach());
  }
  
  // Return update
  // Create a new vector to store the result
  std::vector<torch::Tensor> result;
  result.reserve(model_weights.size());
  
  // Ensure both vectors have the same size
  if (model_weights.size() != w.size()) {
      throw std::runtime_error("Tensor vectors must have the same size for subtraction");
  }
  
  // Perform element-wise subtraction
  for (size_t i = 0; i < model_weights.size(); ++i) {
      result.push_back(model_weights[i] - w[i]);
  }
  
  return result;
}

std::vector<torch::Tensor> MnistTrain::testOG() {
  torch::manual_seed(1);

  SubsetSampler train_sampler = get_subset_sampler(train_dataset_size, subset_size);

  auto train_loader = torch::data::make_data_loader(
    train_dataset,
    train_sampler,
    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));

  auto test_loader = torch::data::make_data_loader(test_dataset, kTestBatchSize);

  torch::optim::SGD optimizer(
    model.parameters(), torch::optim::SGDOptions(learnRate).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, subset_size);
    //test(model, device, *test_loader, test_dataset_size);
  }

  // Extract model weights after training
  std::vector<torch::Tensor> model_weights;
  for (const auto& param : model.parameters()) {
    // Clone the parameter to ensure we're not just storing references
    model_weights.push_back(param.clone().detach());
  }

  return model_weights;
}