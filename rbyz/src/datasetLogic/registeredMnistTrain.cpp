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
#include "logger.hpp"
#include "tensorOps.hpp"

RegisteredMnistTrain::RegisteredMnistTrain(int worker_id, int num_workers, int64_t subset_size)
    : BaseMnistTrain(worker_id, num_workers, subset_size) {
  // Plain mnist to read the images from
  auto plain_mnist = torch::data::datasets::MNIST(kDataRoot);

  SubsetSampler train_sampler = get_subset_sampler(worker_id, DATASET_SIZE, subset_size);
  auto& indices = train_sampler.indices();
  num_samples = indices.size();
  reg_data_size = indices.size() * sample_size;

  // Server will check full forward pass against the clients' first batch of forward pass
  if (worker_id == 0) {
    forward_pass_mem_size = num_samples * forward_pass_info.values_per_sample * forward_pass_info.bytes_per_value;
    forward_pass_indices_mem_size = num_samples * sizeof(uint32_t);
  } else {
    forward_pass_mem_size = kTrainBatchSize * forward_pass_info.values_per_sample * forward_pass_info.bytes_per_value;
    forward_pass_indices_mem_size = kTrainBatchSize * sizeof(uint32_t);
  }
  
  // Init structs
  cudaHostAlloc((void**)&reg_data, reg_data_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&forward_pass, forward_pass_mem_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&forward_pass_indices, forward_pass_indices_mem_size, cudaHostAllocDefault);

  Logger::instance().log("registered_samples: " + std::to_string(indices.size()) + "\n");
  Logger::instance().log("Allocated registered memory for dataset: " +
                         std::to_string(reg_data_size) + " bytes\n");

  // Copy data from the original dataset to the registered memory
  size_t i = 0;  // Counter for the registered memory
  std::unordered_map<size_t, size_t> index_map;
  for (const auto& original_idx : indices) {
    auto example = plain_mnist.get(original_idx);

    // Store idx, for rbyz to identify server VD images
    *getOriginalIndex(i) = static_cast<uint32_t>(original_idx);

    // Copy the label
    *getLabel(i) = static_cast<int64_t>(example.target.item<int64_t>());

    // Copy the image data
    auto normalized_image = (example.data.to(torch::kFloat32) - 0.1307) / 0.3081;
    auto reshaped_image = normalized_image.reshape({1, 28, 28}).contiguous();
    std::memcpy(getImage(i), reshaped_image.data_ptr<float>(), data_info.image_size);

    // Map original index to registered index for retrieval
    index_map[original_idx] = i;
    ++i;
  }

  registered_dataset = std::make_unique<RegisteredMNIST>(data_info, index_map);

  auto loader_temp =
      torch::data::make_data_loader(*registered_dataset,
                                    train_sampler,  // Reuse the same sampler
                                    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
  registered_loader = std::move(loader_temp);

  Logger::instance().log("Registered memory dataset prepared with " +
                         std::to_string(num_samples) + " samples\n");
}

RegisteredMnistTrain::~RegisteredMnistTrain() {
  cudaFreeHost(reg_data);
  cudaFreeHost(forward_pass);
  cudaFreeHost(forward_pass_indices);
  Logger::instance().log("Freed Cuda registered memory for dataset\n");
}

/**
 * Processes batch output and stores per-sample loss and error values efficiently.
 * 
 * @param output Model output tensor (logits)
 * @param targets Ground truth labels
 * @param device Device where tensors reside (CPU/CUDA)
 * @param forward_pass Buffer to store loss and error values
 * @param curr_idx Current index in the forward_pass buffer
 */
void RegisteredMnistTrain::processBatchResults(
    const torch::Tensor& output, 
    const torch::Tensor& targets, 
    torch::Device device,
    float* forward_pass,
    size_t& curr_idx) {
    
  // Calculate loss for each example in the batch at once
  auto individual_losses = torch::nll_loss(output, targets, {}, torch::Reduction::None);
  auto predictions = output.argmax(1);
  auto correct_predictions = predictions.eq(targets);

  // Copy results to CPU if on GPU
  torch::Tensor cpu_losses, cpu_correct;
  if (device.is_cuda()) {
    cpu_losses = torch::empty_like(individual_losses, torch::kCPU);
    cpu_correct = torch::empty_like(correct_predictions, torch::kCPU);
    
    cudaMemcpyAsync(
        cpu_losses.data_ptr<float>(),
        individual_losses.data_ptr<float>(),
        individual_losses.numel() * sizeof(float),
        cudaMemcpyDeviceToHost,
        memcpy_stream_A
    );
    
    cudaMemcpyAsync(
        cpu_correct.data_ptr<bool>(),
        correct_predictions.data_ptr<bool>(),
        correct_predictions.numel() * sizeof(bool),
        cudaMemcpyDeviceToHost,
        memcpy_stream_B
    );
    
    cudaDeviceSynchronize();
  } else {
    cpu_losses = individual_losses;
    cpu_correct = correct_predictions;
  }

  // Copy values to forward_pass buffer
  auto losses_accessor = cpu_losses.accessor<float, 1>();
  auto correct_accessor = cpu_correct.accessor<bool, 1>();

  const size_t forward_pass_size = forward_pass_mem_size / forward_pass_info.bytes_per_value;
  const size_t error_start = forward_pass_size / 2;
  
  for (size_t i = 0; i < cpu_losses.size(0); ++i) {
    if (curr_idx < forward_pass_size / 2) {
      forward_pass[curr_idx] = losses_accessor[i];
      forward_pass[error_start + curr_idx] = correct_accessor[i] ? 0.0f : 1.0f;
      if (curr_idx == 0) Logger::instance().log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      if (curr_idx < 10) {
          Logger::instance().log("Processed sample " + std::to_string(curr_idx) + 
                        " with original index: " + std::to_string(forward_pass_indices[curr_idx]) +
                        " label: " + std::to_string(*getLabel(curr_idx)) +
                        " with loss: " + std::to_string(forward_pass[curr_idx]) + 
                        " and error: " + std::to_string(forward_pass[error_start + curr_idx]) + "\n");
      }

      curr_idx++;
    } else {
      throw std::runtime_error("Forward pass buffer overflow" 
                               " - current index: " + std::to_string(curr_idx) +
                               ", max size: " + std::to_string(forward_pass_size / 2));
    }
  }
}

void RegisteredMnistTrain::train(size_t epoch, 
                           Net& model, 
                           torch::Device device, 
                           torch::optim::Optimizer& optimizer, 
                           size_t dataset_size) {

  Logger::instance().log("  RegTraining model for epoch " + std::to_string(epoch) + "\n");
  model.train();

  size_t batch_idx = 0;
  size_t global_sample_idx = 0;  // Track global sample position

  // For tracking loss and error rate indices in the forward pass buffer
  size_t curr_idx = 0;

  // Track total loss for averaging
  double total_loss = 0.0;
  size_t total_samples = 0;
  size_t total = 0;
  int32_t correct = 0;

  torch::Tensor batch_data = torch::empty({kTrainBatchSize, 1, 28, 28}, torch::kFloat32);
  torch::Tensor batch_targets = torch::empty({kTrainBatchSize}, torch::kInt64);
  
  for (const auto& batch : *registered_loader) { 
    size_t batch_size = batch.size();
  
    // Resize if needed (only for last batch)
    if (batch_size != kTrainBatchSize) {
      batch_data = batch_data.slice(0, 0, batch_size).contiguous();
      batch_targets = batch_targets.slice(0, 0, batch_size).contiguous();
    }
    
    // Direct copy to pre-allocated tensors
    for (size_t i = 0; i < batch_size; ++i) {
      batch_data[i].copy_(batch[i].data);
      batch_targets[i] = batch[i].target;
      forward_pass_indices[global_sample_idx] = *getOriginalIndex(global_sample_idx);
      global_sample_idx++;
    }
    
    torch::Tensor data, targets;
    if (device.is_cuda()) {
      // Create CUDA tensors with the same shape and type
      data = torch::empty_like(batch_data, torch::TensorOptions().device(device));
      targets = torch::empty_like(batch_targets, torch::TensorOptions().device(device));
      
      // Asynchronously copy data from CPU to GPU
      cudaMemcpyAsync(data.data_ptr<float>(),
                     batch_data.data_ptr<float>(),
                     batch_data.numel() * sizeof(float),
                     cudaMemcpyHostToDevice,
                     memcpy_stream_A);
                     
      cudaMemcpyAsync(targets.data_ptr<int64_t>(),
                     batch_targets.data_ptr<int64_t>(),
                     batch_targets.numel() * sizeof(int64_t),
                     cudaMemcpyHostToDevice,
                     memcpy_stream_B);
                     
      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    } else {
      // For CPU, just use the pre-allocated tensors
      data = batch_data;
      targets = batch_targets;
    }

    // Switch to evaluation mode for writing forward pass results
    if (batch_idx == 0 && epoch == 1) {
      model.eval();
      torch::NoGradGuard no_grad;
      auto inference_output = model.forward(data);
      Logger::instance().log("  Going to proc batch results in batch " + std::to_string(batch_idx) + "and epoch " + std::to_string(epoch) + "\n");
      processBatchResults(inference_output, targets, device, forward_pass, curr_idx);
    }

    // Switch back to training mode for training forward pass
    model.train();

    // Clear gradients from previous iteration
    optimizer.zero_grad();

    // Forward pass
    auto output = model.forward(data);

    // Calculate sum of losses (not mean) for this batch
    auto nll_loss_sum = torch::nll_loss(output, targets, {}, torch::Reduction::Sum);
    auto nll_loss_mean = torch::nll_loss(output, targets);  // Keep for backprop
    AT_ASSERT(!std::isnan(nll_loss_mean.template item<float>()));

    // Add batch sum to total loss
    float batch_loss_sum = nll_loss_sum.template item<float>();
    total_loss += batch_loss_sum;
    total_samples += targets.size(0); 

    // Calculate accuracy and error rate for this batch
    auto pred = output.argmax(1);
    int32_t batch_correct = pred.eq(targets).sum().template item<int32_t>();
    correct += batch_correct;
    total += targets.size(0);

    // Backpropagation
    nll_loss_mean.backward();
    optimizer.step();
    
    if (batch_idx % kLogInterval == 0 && worker_id % 2 == 0) {
      std::printf("\rRegTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                 epoch,
                 batch_idx * data.size(0),
                 dataset_size,
                 nll_loss_mean.template item<float>());
    }
    
    batch_idx++;
  }

  // Update the loss member to be the average loss across all batches
  loss = static_cast<float>(total_loss / total_samples);

  // Update the train accuracy member
  setTrainAccuracy(static_cast<float>(correct) / total);

  // Calculate final error rate for the entire epoch (should match 1 - train_accuracy)
  error_rate = 1.0 - (static_cast<double>(correct) / total);

  // Wait for any remaining async operations
  if (device.is_cuda()) {
    cudaDeviceSynchronize();
  }
}

std::vector<torch::Tensor> RegisteredMnistTrain::runMnistTrain(int round, const std::vector<torch::Tensor>& w) {
  std::vector<torch::Tensor> w_cuda = updateModelParameters(w);
  size_t param_count = w_cuda.size();

  //auto test_loader_instance = torch::data::make_data_loader(test_dataset, kTestBatchSize);

  torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(GLOBAL_LEARN_RATE));

  if (round % 1 == 0) {
    std::cout << "Training model for round " << round << " epochs: " << kNumberOfEpochs << "\n";
  }

  Logger::instance().log("Training model for step " + std::to_string(round) + " epochs: " + std::to_string(kNumberOfEpochs) + "\n");

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, optimizer, subset_size);
  }

  if (round % 2 == 0) {
    Logger::instance().log("Testing model after training round " + std::to_string(round) + "\n");
    test(model, device, *test_loader, test_dataset_size);
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
                      cudaMemcpyDeviceToHost,
                      memcpy_stream_A);
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

void RegisteredMnistTrain::runInference(const std::vector<torch::Tensor>& w) {
  torch::NoGradGuard no_grad;  // Prevent gradient calculation
  model.eval();                // Set model to evaluation mode
  updateModelParameters(w);

  updateModelParameters(w);
  Logger::instance().log("  Running inference on registered dataset\n");

  int32_t correct = 0;
  size_t total = 0;
  
  // For tracking loss and error rate indices in the forward pass buffer
  size_t curr_idx = 0;
  size_t global_sample_idx = 0;  // Track global sample position
  
  // Track total loss for averaging
  double total_loss = 0.0;
  size_t total_samples = 0;

  for (const auto& batch : *registered_loader) {
    // Combine all data and targets in the batch into single tensors
    std::vector<torch::Tensor> data_vec, target_vec;
    
    for (const auto& example : batch) {
      data_vec.push_back(example.data);
      target_vec.push_back(example.target);

      forward_pass_indices[global_sample_idx] = *getOriginalIndex(global_sample_idx);
      global_sample_idx++;
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
                      cudaMemcpyHostToDevice,
                      memcpy_stream_A);

      cudaMemcpyAsync(targets.data_ptr<int64_t>(),
                      targets_cpu.data_ptr<int64_t>(),
                      targets_cpu.numel() * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      memcpy_stream_B);

      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    } else {
      // For CPU, just use the original tensors
      data = data_cpu;
      targets = targets_cpu;
    }

    // Run forward pass
    auto output = model.forward(data);
    
    // Process batch results
    processBatchResults(output, targets, device, forward_pass, curr_idx);
    
    // Calculate sum of losses (not mean) for this batch
    auto nll_loss_sum = torch::nll_loss(output, targets, {}, torch::Reduction::Sum);
    float batch_loss_sum = nll_loss_sum.template item<float>();
    total_loss += batch_loss_sum;
    total_samples += targets.size(0);  // Add number of samples in this batch
    
    // Calculate accuracy for this batch
    auto pred = output.argmax(1);
    int32_t batch_correct = pred.eq(targets).sum().template item<int32_t>();
    correct += batch_correct;
    total += targets.size(0);
  }

  // Calculate true mean loss across all samples
  loss = static_cast<float>(total_loss / total_samples);
  
  // Calculate final error rate for the entire inference (should match 1 - accuracy)
  error_rate = 1.0 - (static_cast<double>(correct) / total);
  
  // Wait for any remaining async operations
  if (device.is_cuda()) {
    cudaDeviceSynchronize();
  }
  
  Logger::instance().log("Inference completed - Loss: " + std::to_string(loss) + 
                        ", Error rate: " + std::to_string(error_rate) + 
                        ", Total samples: " + std::to_string(total) + "\n");
}