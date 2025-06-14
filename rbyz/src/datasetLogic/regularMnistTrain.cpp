#include "datasetLogic/regularMnistTrain.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"

RegularMnistTrain::RegularMnistTrain(int worker_id, int num_workers, int64_t subset_size)
    : BaseMnistTrain(worker_id, num_workers, subset_size),
      train_dataset(torch::data::datasets::MNIST(kDataRoot)
                        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                        .map(torch::data::transforms::Stack<>())) {
              
  size_t train_dataset_size = train_dataset.size().value();

  SubsetSampler train_sampler = get_subset_sampler(worker_id, train_dataset_size, subset_size);
  auto train_loader_temp = torch::data::make_data_loader(
      train_dataset, train_sampler, torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  train_loader = std::move(train_loader_temp);
}

template <typename DataLoader>
void RegularMnistTrain::train(size_t epoch,
                              Net& model,
                              torch::Device device,
                              DataLoader& data_loader,
                              torch::optim::Optimizer& optimizer,
                              size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  int32_t correct = 0;
  size_t total = 0;
  
  // Track total loss for averaging
  double total_loss = 0.0;
  size_t total_batches = 0;
  
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
                                  10); // Print at most 10 elements
      for (int i = 0; i < num_to_print; ++i) {
        oss << targets_device[i].item<int64_t>();
        if (i < num_to_print - 1)
          oss << ", ";
      }
      oss << "]";
      Logger::instance().log(oss.str() + "\n");
    }

    optimizer.zero_grad();
    auto output = model.forward(data_device);
    auto nll_loss = torch::nll_loss(output, targets_device);
    AT_ASSERT(!std::isnan(nll_loss.template item<float>()));
    
    // Track batch loss for averaging later
    float batch_loss = nll_loss.template item<float>();
    total_loss += batch_loss * targets_device.size(0); // Weight by batch size
    total_batches += targets_device.size(0);

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
                  batch_loss); // Show current batch loss in progress messages
    }
  }
  
  // Set the loss member to the average loss across all batches
  loss = static_cast<float>(total_loss / total_batches);
}

std::vector<torch::Tensor> RegularMnistTrain::runMnistTrain(int round,
                                                            const std::vector<torch::Tensor>& w) {
  // Update model parameters, w is in cpu so if device is cuda copy to device is needed
  std::vector<torch::Tensor> w_cuda = updateModelParameters(w);
  size_t param_count = w_cuda.size();

  torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(GLOBAL_LEARN_RATE));

  std::cout << "Training model for round " << round << " epochs: " << kNumberOfEpochs << "\n";

  if (round % 2 == 0) {
    Logger::instance().log("Testing model pre training round " + std::to_string(round) + "\n");
    test(model, device, *test_loader, test_dataset_size);
  }

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, subset_size);
  }

  std::vector<torch::Tensor> params = model.parameters();
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

void RegularMnistTrain::runInference(const std::vector<torch::Tensor>& w) {
  torch::NoGradGuard no_grad;  // Prevent gradient calculation
  model.eval();                // Set model to evaluation mode
  updateModelParameters(w);

  throw std::runtime_error("runInference not fully implemented yet for RegularMnistTrain");

  int32_t correct = 0;
  size_t total = 0;
  double total_loss = 0.0;

  for (auto& batch : *train_loader) {
    // Copy to GPU if needed for faster performance
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);

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