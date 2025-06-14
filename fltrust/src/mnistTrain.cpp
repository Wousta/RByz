#include "mnistTrain.hpp"
#include "globalConstants.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <unordered_map>
#include <algorithm>
#include <random>

MnistTrain::MnistTrain(int worker_id, int num_workers, int64_t subset_size)
    : device(init_device()),
      worker_id(worker_id),
      num_workers(num_workers),
      subset_size(subset_size),
      train_dataset(torch::data::datasets::MNIST(kDataRoot)
                        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                        .map(torch::data::transforms::Stack<>())),
      test_dataset(torch::data::datasets::MNIST(
                       kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                       .map(torch::data::transforms::Stack<>()))
{
  torch::manual_seed(1);
  model.to(device);

  train_dataset_size = train_dataset.size().value();
  test_dataset_size = test_dataset.size().value();

  SubsetSampler train_sampler = get_subset_sampler(worker_id, train_dataset_size, subset_size);
  auto train_loader_temp = torch::data::make_data_loader(
      train_dataset,
      train_sampler,
      torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(train_loader_temp);

  // Create test_loader with std::make_unique
  auto test_loader_temp = torch::data::make_data_loader(
      test_dataset,
      torch::data::DataLoaderOptions().batch_size(kTestBatchSize));
  test_loader = std::move(test_loader_temp);
}

torch::Device MnistTrain::init_device()
{
  try
  {
    if (torch::cuda::is_available())
    {
      std::cout << "CUDA is available, using GPU" << std::endl;
      return torch::Device(torch::kCUDA);
    }
    else
    {
      std::cout << "CUDA is not available, using CPU" << std::endl;
      return torch::Device(torch::kCPU);
    }
  }
  catch (const c10::Error &e)
  {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    std::cout << "Falling back to CPU" << std::endl;
    return torch::Device(torch::kCPU);
  }
}

SubsetSampler MnistTrain::get_subset_sampler(int worker_id, size_t dataset_size, int64_t subset_size)
{
  auto stratified_indices = get_stratified_indices(train_dataset, worker_id, num_workers, subset_size);

  Logger::instance().log("Worker " + std::to_string(worker_id) +
                         " using stratified indices of size: " + std::to_string(stratified_indices.size()) + "\n");

  return SubsetSampler(stratified_indices);
}

std::vector<size_t> MnistTrain::get_stratified_indices(
  MnistTrain::DatasetType& dataset,
  int worker_id,
  int num_workers,
  size_t subset_size) 
  {
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
          start = 0; // Server gets the first portion
          end = samples_srvr;
      } else {
          start = static_cast<size_t>(std::ceil(samples_srvr + (worker_id - 1) * samples_clnt));
          if (worker_id == num_workers - 1) {
            end = total_samples; // Last worker gets the remaining samples
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

  // Log the final indices
  Logger::instance().log("Worker " + std::to_string(worker_id) +
                         " final indices size: " + std::to_string(worker_indices.size()) + "\n");

  return worker_indices;
}

std::vector<torch::Tensor> MnistTrain::runMnistTrainDummy(std::vector<torch::Tensor> &w)
{
  std::cout << "Running dummy MNIST training\n";

  for (size_t i = 0; i < w.size(); i++)
  {
    w[i] = w[i] + 1; // element-wise addition of 1
  }

  return w;
}

template <typename DataLoader>
void MnistTrain::train(
    size_t epoch,
    Net &model,
    torch::Device device,
    DataLoader &data_loader,
    torch::optim::Optimizer &optimizer,
    size_t dataset_size)
{
  model.train();
  size_t batch_idx = 0;
  int32_t correct = 0;
  size_t total = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for (auto &batch : data_loader)
  {
    torch::Tensor data_device, targets_device;

    if (device.is_cuda())
    {
      // Create CUDA tensors with the same shape and type
      data_device = torch::empty_like(batch.data, torch::TensorOptions().device(device));
      targets_device = torch::empty_like(batch.target, torch::TensorOptions().device(device));

      // Asynchronously copy data from CPU to GPU
      cudaMemcpyAsync(
          data_device.data_ptr<float>(),
          batch.data.template data_ptr<float>(),
          batch.data.numel() * sizeof(float),
          cudaMemcpyHostToDevice);

      cudaMemcpyAsync(
          targets_device.data_ptr<int64_t>(),
          batch.target.template data_ptr<int64_t>(),
          batch.target.numel() * sizeof(int64_t),
          cudaMemcpyHostToDevice);

      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    }
    else
    {
      // For CPU, just use the original tensors
      data_device = batch.data;
      targets_device = batch.target;
    }

    optimizer.zero_grad();
    output = model.forward(data_device);
    
    // Calculate sum of losses (not mean) for this batch
    auto nll_loss_mean = torch::nll_loss(output, targets_device);
    AT_ASSERT(!std::isnan(nll_loss_mean.template item<float>()));

    // Backpropagation (use mean loss for gradients)
    nll_loss_mean.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0 && worker_id % 2 == 0)
    {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          nll_loss_mean.template item<float>());
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
    Logger::instance().log("Total time taken server side step: " +
                          std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) +
                          " ms\n");
}

template <typename DataLoader>
void MnistTrain::test(
    Net &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size)
{
  torch::NoGradGuard no_grad;
  model.eval();
  
  double total_loss = 0.0;
  int32_t correct = 0;
  int32_t total_samples = 0;  // Track actual samples processed
  
  for (const auto &batch : data_loader)
  {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    
    // Calculate sum of losses for this batch
    total_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
    total_samples += targets.size(0);  // Add actual batch size
  }

  // Update class members with properly calculated metrics
  loss = static_cast<float>(total_loss / total_samples);  // True mean loss
  error_rate = 1.0 - (static_cast<float>(correct) / total_samples);  // True error rate
  accuracy = static_cast<float>(correct) / total_samples;  // True accuracy
  
  std::ostringstream oss;
  oss << "\nTest set: Average loss: " << std::fixed << std::setprecision(4) << loss
      << " | Accuracy: " << std::fixed << std::setprecision(3)
      << static_cast<double>(correct) / total_samples
      << " | Error rate: " << std::fixed << std::setprecision(3) << error_rate
      << " (" << correct << "/" << total_samples << ")";  // Show actual counts
  Logger::instance().log(oss.str());
  Logger::instance().log("Testing done\n");
}

std::vector<torch::Tensor> MnistTrain::runMnistTrain(int round, const std::vector<torch::Tensor> &w)
{
  // Update model parameters, w is in cpu so if device is cuda copy to device is needed
  std::vector<torch::Tensor> params = model.parameters();
  size_t param_count = model.parameters().size();
  std::vector<torch::Tensor> w_cuda;
  if (device.is_cuda())
  {
    w_cuda.reserve(param_count);
    for (const auto &param : w)
    {
      auto cuda_tensor = torch::empty_like(param, torch::kCUDA);
      cudaMemcpyAsync(
          cuda_tensor.data_ptr<float>(),
          param.data_ptr<float>(),
          param.numel() * sizeof(float),
          cudaMemcpyHostToDevice);
      w_cuda.push_back(cuda_tensor);
    }

    cudaDeviceSynchronize();
    for (size_t i = 0; i < params.size(); ++i)
    {
      params[i].data().copy_(w_cuda[i]);
    }
  }
  else
  {
    for (size_t i = 0; i < params.size(); ++i)
    {
      params[i].data().copy_(w[i]);
    }
  }

  Logger::instance().log("PRE TEST ACC:\n");
  test(model, device, *test_loader, test_dataset_size);

  auto test_loader = torch::data::make_data_loader(test_dataset, kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(GLOBAL_LEARN_RATE));

  std::cout << "Training model for round " << round << " epochs: " << kNumberOfEpochs << "\n";
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
  {
    train(epoch, model, device, *train_loader, optimizer, subset_size);
    // train(epoch, model, device, *train_loader_default, optimizer, train_dataset_size_default);
  }

  // if (round % 2 == 0)
  // {
  //   Logger::instance().log("Testing model after training round " + std::to_string(round) + "\n");
  //   test(model, device, *test_loader, test_dataset_size);
  // }

  std::vector<torch::Tensor> result;
  result.reserve(param_count);
  if (device.is_cuda())
  {
    for (size_t i = 0; i < params.size(); ++i)
    {
      // Subtract on GPU
      auto update = params[i].clone().detach() - w_cuda[i];

      // Copy result to CPU
      auto cpu_update = torch::empty_like(update, torch::kCPU);
      cudaMemcpyAsync(
          cpu_update.data_ptr<float>(),
          update.data_ptr<float>(),
          update.numel() * sizeof(float),
          cudaMemcpyDeviceToHost);
      result.push_back(cpu_update);
    }
    cudaDeviceSynchronize();
  }
  else
  {
    std::vector<torch::Tensor> model_weights;
    model_weights.reserve(param_count);
    for (const auto &param : model.parameters()) {
      model_weights.push_back(param.clone().detach());
    }
    for (size_t i = 0; i < model_weights.size(); ++i) {
      result.push_back(model_weights[i] - w[i]);
    }
  }

  return result;
}

std::vector<torch::Tensor> MnistTrain::updateModelParameters(const std::vector<torch::Tensor>& w) {
  // Update model parameters, w is in cpu so if device is cuda copy to device is needed
  std::vector<torch::Tensor> params = model.parameters();
  size_t param_count = params.size();
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

    return w_cuda; // Return CUDA tensors if device is CUDA
  } else {
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w[i]);
    }

    return w; // Return CPU tensors if device is CPU
  }
}

std::vector<torch::Tensor> MnistTrain::getInitialWeights()
{
  // Get model's parameters
  std::vector<torch::Tensor> initialWeights;
  for (const auto &param : model.parameters())
  {
    // Clone and detach parameters to CPU tensors
    initialWeights.push_back(param.clone().detach().to(torch::kCPU));
  }

  Logger::instance().log("Obtained initial random weights from model\n");
  printTensorSlices(initialWeights, 0, 5);

  return initialWeights;
}

void MnistTrain::testModel()
{
  test(model, device, *test_loader, test_dataset_size);
}