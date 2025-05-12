#include "../include/mnistTrain.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/globalConstants.hpp"
#include "../include/logger.hpp"
#include "../include/tensorOps.hpp"

MnistTrain::MnistTrain(int worker_id, int num_workers, int64_t subset_size)
    : device(init_device()),
      worker_id(worker_id),
      num_workers(num_workers),
      subset_size(subset_size),
      train_dataset(torch::data::datasets::MNIST(kDataRoot)
                        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                        .map(torch::data::transforms::Stack<>())),
      test_dataset(
          torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
              .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
              .map(torch::data::transforms::Stack<>())) {
  torch::manual_seed(1);
  model.to(device);

  train_dataset_size = train_dataset.size().value();
  test_dataset_size = test_dataset.size().value();

  SubsetSampler train_sampler = get_subset_sampler(worker_id, train_dataset_size, subset_size);
  auto train_loader_temp = torch::data::make_data_loader(
      train_dataset, train_sampler, torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(train_loader_temp);

  // Create test_loader with std::make_unique
  auto test_loader_temp = torch::data::make_data_loader(
      test_dataset, torch::data::DataLoaderOptions().batch_size(kTestBatchSize));
  test_loader = std::move(test_loader_temp);

  // Dataset registered in memory for RByz validation dataset remote placement
  prepareRegisteredDataset();
}

torch::Device MnistTrain::init_device() {
  try {
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available, using GPU" << std::endl;
      return torch::Device(torch::kCUDA);
    } else {
      std::cout << "CUDA is not available, using CPU" << std::endl;
      return torch::Device(torch::kCPU);
    }
  } catch (const c10::Error& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    std::cout << "Falling back to CPU" << std::endl;
    return torch::Device(torch::kCPU);
  }
}

SubsetSampler MnistTrain::get_subset_sampler(int worker_id,
                                             size_t dataset_size,
                                             int64_t subset_size) {
  auto stratified_indices =
      get_stratified_indices(train_dataset, worker_id, num_workers, subset_size);

  Logger::instance().log(
      "Worker " + std::to_string(worker_id) +
      " using stratified indices of size: " + std::to_string(stratified_indices.size()) + "\n");

  return SubsetSampler(stratified_indices);
}

std::vector<size_t> MnistTrain::get_stratified_indices(MnistTrain::DatasetType& dataset,
                                                       int worker_id,
                                                       int num_workers,
                                                       size_t subset_size) {
  // Group indices by labels
  std::unordered_map<int64_t, std::vector<size_t>> label_to_indices;
  size_t index = 0;

  // Create a DataLoader to iterate over the dataset
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      dataset, /*batch size*/ 1);

  for (const auto& example : *data_loader) {
    int64_t label = example.target.item<int64_t>();
    label_to_indices[label].push_back(index);
    ++index;
  }

  // Allocate indices to workers
  float srvr_proportion = static_cast<float>(SRVR_SUBSET_SIZE) / DATASET_SIZE;
  float clnt_proportion = (1 - srvr_proportion) / (num_workers - 1);

  std::vector<size_t> worker_indices;
  for (const auto& [label, indices] : label_to_indices) {
    size_t total_samples = indices.size();
    size_t samples_srvr = static_cast<size_t>(std::ceil(total_samples * srvr_proportion));
    size_t samples_clnt = static_cast<size_t>(std::floor(total_samples * clnt_proportion));

    size_t start;
    size_t end;
    if (worker_id == 0) {
      start = 0;  // Server gets the first portion
      end = samples_srvr;
    } else {
      start = static_cast<size_t>(std::ceil(samples_srvr + (worker_id - 1) * samples_clnt));
      if (worker_id == num_workers - 1) {
        end = total_samples;  // Last worker gets the remaining samples
      } else {
        end = start + samples_clnt;
      }
    }

    // Add the worker's portion of indices for this label
    worker_indices.insert(worker_indices.end(), indices.begin() + start, indices.begin() + end);
  }

  std::random_device rd;
  std::mt19937 rng(rd());
  std::shuffle(worker_indices.begin(), worker_indices.end(), rng);

  // If the subset size is smaller than the allocated indices, truncate
  if (worker_indices.size() > subset_size) {
    worker_indices.resize(subset_size);
  }

  return worker_indices;
}

template <typename DataLoader>
void MnistTrain::train(size_t epoch,
                       Net& model,
                       torch::Device device,
                       DataLoader& data_loader,
                       torch::optim::Optimizer& optimizer,
                       size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  int32_t correct = 0;
  size_t total = 0;
  for (auto& batch : data_loader) {
    torch::Tensor data_device, targets_device;

    if (device.is_cuda()) {
      // Create CUDA tensors with the same shape and type
      data_device = torch::empty_like(batch.data, torch::TensorOptions().device(device));
      targets_device = torch::empty_like(batch.target, torch::TensorOptions().device(device));

      // Asynchronously copy data from CPU to GPU
      cudaMemcpyAsync(data_device.data_ptr<float>(),
                      batch.data.template data_ptr<float>(),
                      batch.data.numel() * sizeof(float),
                      cudaMemcpyHostToDevice);

      cudaMemcpyAsync(targets_device.data_ptr<int64_t>(),
                      batch.target.template data_ptr<int64_t>(),
                      batch.target.numel() * sizeof(int64_t),
                      cudaMemcpyHostToDevice);

      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    } else {
      // For CPU, just use the original tensors
      data_device = batch.data;
      targets_device = batch.target;
    }

    if (batch_idx == 0) {
      std::ostringstream oss;
      oss << "  Targets (first 10 elements): [";
      int num_to_print = std::min(static_cast<int>(targets_device.numel()),
                                  static_cast<int>(targets_device.numel()));
      for (int i = 0; i < num_to_print; ++i) {
        oss << targets_device[i].item<int64_t>();
        if (i < num_to_print - 1)
          oss << ", ";
      }
      oss << "]";
      Logger::instance().log(oss.str() + "\n");
    }

    optimizer.zero_grad();
    output = model.forward(data_device);
    auto nll_loss = torch::nll_loss(output, targets_device);
    AT_ASSERT(!std::isnan(nll_loss.template item<float>()));
    loss = nll_loss.template item<float>();

    // Calculate accuracy and error rate for this batch
    auto pred = output.argmax(1);
    int32_t batch_correct = pred.eq(targets_device).sum().template item<int32_t>();
    correct += batch_correct;
    total += targets_device.size(0);
    error_rate = 1.0 - (static_cast<float>(correct) / total);

    // Backpropagation
    nll_loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0 && worker_id % 2 == 0) {
      std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                  epoch,
                  batch_idx * batch.data.size(0),
                  dataset_size,
                  nll_loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void MnistTrain::test(Net& model,
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
    test_loss += torch::nll_loss(output,
                                 targets,
                                 /*weight=*/{},
                                 torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::ostringstream oss;
  oss << "\nTest set: Average loss: " << std::fixed << std::setprecision(4) << test_loss
      << " | Accuracy: " << std::fixed << std::setprecision(3)
      << static_cast<double>(correct) / dataset_size << std::endl;
  Logger::instance().log(oss.str());
  Logger::instance().log("Testing done\n");
}

std::vector<torch::Tensor> MnistTrain::runMnistTrain(int round,
                                                     const std::vector<torch::Tensor>& w,
                                                     bool registered_mode) {
  // Update model parameters, w is in cpu so if device is cuda copy to device is needed
  std::vector<torch::Tensor> params = model.parameters();
  size_t param_count = model.parameters().size();
  std::vector<torch::Tensor> w_cuda;
  if (device.is_cuda()) {
    w_cuda.reserve(param_count);
    for (const auto& param : w) {
      auto cuda_tensor = torch::empty_like(param, torch::kCUDA);
      cudaMemcpyAsync(cuda_tensor.data_ptr<float>(),
                      param.data_ptr<float>(),
                      param.numel() * sizeof(float),
                      cudaMemcpyHostToDevice);
      w_cuda.push_back(cuda_tensor);
    }

    cudaDeviceSynchronize();
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w_cuda[i]);
    }

  } else {
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w[i]);
    }
  }

  auto test_loader = torch::data::make_data_loader(test_dataset, kTestBatchSize);

  torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(GLOBAL_LEARN_RATE));

  std::cout << "Training model for round " << round << " epochs: " << kNumberOfEpochs << "\n";

  if (registered_mode) {
    // Use registered dataset
    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
      registeredTrain(epoch, model, device, optimizer, registered_samples);
    }
  } else {
    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
      train(epoch, model, device, *train_loader, optimizer, subset_size);
    }
  }

  if (round % 2 == 0) {
    Logger::instance().log("Testing model after training round " + std::to_string(round) + "\n");
    test(model, device, *test_loader, test_dataset_size);
  }

  std::vector<torch::Tensor> result;
  result.reserve(param_count);
  if (device.is_cuda()) {
    for (size_t i = 0; i < params.size(); ++i) {
      // Subtract on GPU
      auto update = params[i].clone().detach() - w_cuda[i];

      // Copy result to CPU
      auto cpu_update = torch::empty_like(update, torch::kCPU);
      cudaMemcpyAsync(cpu_update.data_ptr<float>(),
                      update.data_ptr<float>(),
                      update.numel() * sizeof(float),
                      cudaMemcpyDeviceToHost);
      result.push_back(cpu_update);
    }
    cudaDeviceSynchronize();
  } else {
    std::vector<torch::Tensor> model_weights;
    model_weights.reserve(param_count);
    for (const auto& param : model.parameters()) {
      model_weights.push_back(param.clone().detach());
    }
    for (size_t i = 0; i < model_weights.size(); ++i) {
      result.push_back(model_weights[i] - w[i]);
    }
  }

  return result;
}

std::vector<torch::Tensor> MnistTrain::getInitialWeights() {
  // Get model's parameters
  std::vector<torch::Tensor> initialWeights;
  for (const auto& param : model.parameters()) {
    // Clone and detach parameters to CPU tensors
    initialWeights.push_back(param.clone().detach().to(torch::kCPU));
  }

  Logger::instance().log("Obtained initial random weights from model\n");
  printTensorSlices(initialWeights, 0, 5);

  return initialWeights;
}

void MnistTrain::testModel() {
  test(model, device, *test_loader, test_dataset_size);
}

// Updates the error and loss of this model
void MnistTrain::runInference() {
  torch::NoGradGuard no_grad;  // Prevent gradient calculation
  model.eval();                // Set model to evaluation mode

  int32_t correct = 0;
  size_t total = 0;
  double total_loss = 0.0;

  // TODO: It is still using the non registered train_loader
  for (auto& batch : *train_loader) {
    // Copy to gpu here for faster performance
    auto data = batch.data.to(device), targets = batch.target.to(device);
    output = model.forward(data);

    auto nll_loss = torch::nll_loss(output, targets);
    total_loss += nll_loss.template item<float>() * targets.size(0);

    // Calculate accuracy and error rate for this batch
    auto pred = output.argmax(1);
    int32_t batch_correct = pred.eq(targets).sum().template item<int32_t>();
    correct += batch_correct;
    total += targets.size(0);
  }

  loss = total_loss / total;
  error_rate = 1.0 - (static_cast<float>(correct) / total);
}

// Saves the model parameters into a .pt file. Used to avoid having to run FLtrust when testing
void MnistTrain::saveModelState(const std::vector<torch::Tensor>& w, const std::string& filename) {
  try {
    Logger::instance().log("Saving model state to " + filename + "...\n");
    torch::save(w, filename);
    Logger::instance().log("Model state saved successfully.\n");
  } catch (const std::exception& e) {
    Logger::instance().log("Error saving model state: " + std::string(e.what()) + "\n");
  }
}

// Loads the model parameters from a .pt file and returns them. Used to avoid having to run FLtrust
// when testing
std::vector<torch::Tensor> MnistTrain::loadModelState(const std::string& filename) {
  std::vector<torch::Tensor> w;
  try {
    Logger::instance().log("Loading model state from " + filename + "...\n");
    torch::load(w, filename);
    Logger::instance().log("Model state loaded successfully.\n");
  } catch (const std::exception& e) {
    Logger::instance().log("Error loading model state: " + std::string(e.what()) + "\n");
  }
  return w;
}

void MnistTrain::prepareRegisteredDataset() {
  auto plain_mnist = torch::data::datasets::MNIST(kDataRoot);

  SubsetSampler train_sampler = get_subset_sampler(worker_id, train_dataset_size, subset_size);
  auto& indices = train_sampler.indices();
  registered_samples = indices.size();

  // 28x28 = 784 is the size of an image in MNIST dataset
  size_t image_memory_size = registered_samples * 784 * sizeof(float);
  size_t label_memory_size = registered_samples * sizeof(int64_t);

  registered_images = reinterpret_cast<float*>(malloc(image_memory_size));
  registered_labels = reinterpret_cast<int64_t*>(malloc(label_memory_size));

  Logger::instance().log("registered_samples: " + std::to_string(registered_samples) + "\n");
  Logger::instance().log("Allocated registered memory for dataset: " +
                         std::to_string(image_memory_size + label_memory_size) + " bytes\n");

  // Copy data from the original dataset to the registered memory
  size_t i = 0;  // Counter for the registered memory
  std::unordered_map<size_t, size_t> index_map;
  for (const auto& original_idx : indices) {
    auto example = plain_mnist.get(original_idx);
    auto normalized_image = (example.data.to(torch::kFloat32) - 0.1307) / 0.3081;

    // Flatten the image tensor and copy it to the registered memory
    auto reshaped_image = normalized_image.reshape({1, 28, 28}).contiguous();
    std::memcpy(
        registered_images + (i * 784), reshaped_image.data_ptr<float>(), 784 * sizeof(float));

    // Copy the label
    registered_labels[i] = example.target.item<int64_t>();

    // Validate the label
    if (registered_labels[i] < 0 || registered_labels[i] >= 10) {
      throw std::runtime_error("Invalid label: " + std::to_string(registered_labels[i]));
    }

    // Map original index to registered index
    index_map[original_idx] = i;
    ++i;
  }

  registered_dataset = std::make_unique<RegisteredMNIST>(
      registered_images, registered_labels, registered_samples, index_map, 784, false);

  auto loader_temp =
      torch::data::make_data_loader(*registered_dataset,
                                    train_sampler,  // Reuse the same sampler
                                    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  registered_loader = std::move(loader_temp);

  Logger::instance().log("Registered memory dataset prepared with " +
                         std::to_string(registered_samples) + " samples\n");
}

void MnistTrain::registeredTrain(size_t epoch,
                                 Net& model,
                                 torch::Device device,
                                 torch::optim::Optimizer& optimizer,
                                 size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  int32_t correct = 0;
  size_t total = 0;

  for (const auto& batch : *registered_loader) {
    // Combine all data and targets in the batch into single tensors
    std::vector<torch::Tensor> data_vec, target_vec;
    for (const auto& example : batch) {
      data_vec.push_back(example.data);
      target_vec.push_back(example.target);
    }

    // Stack the tensors to create a single batch tensor
    auto data_cpu = torch::stack(data_vec);
    auto targets_cpu = torch::stack(target_vec);

    torch::Tensor data, targets;
    if (device.is_cuda()) {
      // Create CUDA tensors with the same shape and type
      data = torch::empty_like(data_cpu, torch::TensorOptions().device(device));
      targets = torch::empty_like(targets_cpu, torch::TensorOptions().device(device));

      // Asynchronously copy data from CPU to GPU
      cudaMemcpyAsync(data.data_ptr<float>(),
                      data_cpu.data_ptr<float>(),
                      data_cpu.numel() * sizeof(float),
                      cudaMemcpyHostToDevice);

      cudaMemcpyAsync(targets.data_ptr<int64_t>(),
                      targets_cpu.data_ptr<int64_t>(),
                      targets_cpu.numel() * sizeof(int64_t),
                      cudaMemcpyHostToDevice);

      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    } else {
      // For CPU, just use the original tensors
      data = data_cpu;
      targets = targets_cpu;
    }

    optimizer.zero_grad();
    auto output = model.forward(data);
    auto nll_loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(nll_loss.template item<float>()));
    loss = nll_loss.template item<float>();

    // Calculate accuracy and error rate for this batch
    auto pred = output.argmax(1);
    int32_t batch_correct = pred.eq(targets).sum().template item<int32_t>();
    correct += batch_correct;
    total += targets.size(0);
    error_rate = 1.0 - (static_cast<float>(correct) / total);

    // Backpropagation
    nll_loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0 && worker_id % 2 == 0) {
      std::printf("\rRegTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                  epoch,
                  batch_idx * data.size(0),
                  dataset_size,
                  nll_loss.template item<float>());
    }
  }
}
