#include "datasetLogic/registeredMnistTrain.hpp"

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
#include "global/logger.hpp"
#include "tensorOps.hpp"

RegisteredMnistTrain::RegisteredMnistTrain(int worker_id, int num_workers, int64_t subset_size)
    : BaseMnistTrain(worker_id, num_workers, subset_size) {
  // Plain mnist to read the images from
  auto plain_mnist = torch::data::datasets::MNIST(kDataRoot);

  SubsetSampler train_sampler = get_subset_sampler(worker_id, train_dataset_size, subset_size);
  auto& indices = train_sampler.indices();
  registered_samples = indices.size();

  // For MNIST forward pass: 10 output values (logits) per sample
  const size_t values_per_sample = 10;
  const size_t bytes_per_value = sizeof(float);

  // 28x28 = 784 is the size of an image in MNIST dataset
  images_mem_size = registered_samples * data_size * sizeof(float);
  labels_mem_size = registered_samples * sizeof(int64_t);
  forward_pass_mem_size = registered_samples * values_per_sample * bytes_per_value;
  forward_pass_indices_mem_size = registered_samples * sizeof(int32_t);
  registered_images = reinterpret_cast<float*>(malloc(images_mem_size));
  registered_labels = reinterpret_cast<int64_t*>(malloc(labels_mem_size));
  forward_pass = reinterpret_cast<float*>(malloc(forward_pass_mem_size));
  forward_pass_indices = reinterpret_cast<int32_t*>(malloc(forward_pass_indices_mem_size));
  if (!registered_images || !registered_labels || !forward_pass || !forward_pass_indices) {
    throw std::runtime_error("Failed to allocate memory for registered dataset");
  }

  Logger::instance().log("registered_samples: " + std::to_string(registered_samples) + "\n");
  Logger::instance().log("Allocated registered memory for dataset: " +
                         std::to_string(images_mem_size + labels_mem_size) + " bytes\n");

  // Copy data from the original dataset to the registered memory
  size_t i = 0;  // Counter for the registered memory
  std::unordered_map<size_t, size_t> index_map;
  for (const auto& original_idx : indices) {
    auto example = plain_mnist.get(original_idx);
    auto normalized_image = (example.data.to(torch::kFloat32) - 0.1307) / 0.3081;

    // Flatten the image tensor and copy it to the registered memory
    auto reshaped_image = normalized_image.reshape({1, 28, 28}).contiguous();
    std::memcpy(
        registered_images + (i * data_size), reshaped_image.data_ptr<float>(), 784 * sizeof(float));

    // Store original_idx in the last position, for rbyz to identify server VD images
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(&registered_images[i * data_size + 784]);
    *index_ptr = static_cast<uint32_t>(original_idx);

    // Copy the label
    registered_labels[i] = example.target.item<int64_t>();
    if (registered_labels[i] < 0 || registered_labels[i] >= 10) {
      throw std::runtime_error("Invalid label: " + std::to_string(registered_labels[i]));
    }

    // Map original index to registered index for retrieval
    index_map[original_idx] = i;
    ++i;
  }

  registered_dataset = std::make_unique<RegisteredMNIST>(
      registered_images, registered_labels, registered_samples, index_map, data_size, false);

  auto loader_temp =
      torch::data::make_data_loader(*registered_dataset,
                                    train_sampler,  // Reuse the same sampler
                                    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  registered_loader = std::move(loader_temp);

  Logger::instance().log("Registered memory dataset prepared with " +
                         std::to_string(registered_samples) + " samples\n");
}

RegisteredMnistTrain::~RegisteredMnistTrain() {
  free(registered_images);
  free(registered_labels);
  free(forward_pass);
  free(forward_pass_indices);
  Logger::instance().log("Freed registered memory for dataset\n");
}

void RegisteredMnistTrain::train(size_t epoch, 
                            Net& model, 
                            torch::Device device, 
                            torch::optim::Optimizer& optimizer, 
                            size_t dataset_size) {

  Logger::instance().log("Training model for epoch " + std::to_string(epoch) + "\n");
  model.train();
  size_t batch_idx = 0;
  int32_t correct = 0;
  size_t total = 0;
  uint32_t img_idx = 0;
  size_t start_pos_output = 0;

  for (const auto& batch : *registered_loader) {
    // Combine all data and targets in the batch into single tensors
    std::vector<torch::Tensor> data_vec, target_vec;
    data_vec.reserve(batch.size());
    target_vec.reserve(batch.size());
    for (const auto& example : batch) {
      data_vec.push_back(example.data);
      target_vec.push_back(example.target);
      uint32_t original_idx = getOriginalIndex(img_idx);

      // Copy the data to registered memory
      forward_pass_indices[img_idx] = original_idx;
      ++img_idx;
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
    size_t output_elements = output.numel();

    // Copy the output to registered memory
    std::memcpy(forward_pass + start_pos_output, 
                output.data_ptr<float>(),
                output_elements * sizeof(float));
    start_pos_output += output_elements;

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

std::vector<torch::Tensor> RegisteredMnistTrain::runMnistTrain(int round, const std::vector<torch::Tensor>& w) {
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
  } else {
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w[i]);
    }
  }

  //auto test_loader_instance = torch::data::make_data_loader(test_dataset, kTestBatchSize);

  torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(GLOBAL_LEARN_RATE));

  std::cout << "Training model for round " << round << " epochs: " << kNumberOfEpochs << "\n";

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, optimizer, subset_size);
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

void RegisteredMnistTrain::runInference() {
  torch::NoGradGuard no_grad;  // Prevent gradient calculation
  model.eval();                // Set model to evaluation mode

  int32_t correct = 0;
  size_t total = 0;
  double total_loss = 0.0;
  int batch_idx = 0;

  for (const auto& batch : *registered_loader) {
    batch_idx++;
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
    
    // Run forward pass
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
