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
#include <cassert>

#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"

RegisteredMnistTrain::RegisteredMnistTrain(int worker_id, int num_workers, int64_t subset_size)
    : BaseMnistTrain(worker_id, num_workers, subset_size) {

  auto plain_mnist = torch::data::datasets::MNIST(kDataRoot);
  SubsetSampler train_sampler = get_subset_sampler(worker_id, DATASET_SIZE, subset_size);
  auto& indices = train_sampler.indices();

  // Initialize data info
  num_samples = indices.size();
  reg_data_size = indices.size() * sample_size;
  forward_pass_mem_size = num_samples * forward_pass_info.values_per_sample * forward_pass_info.bytes_per_value;
  forward_pass_indices_mem_size = num_samples * sizeof(uint32_t);
  forward_pass_size = forward_pass_mem_size / forward_pass_info.bytes_per_value;
  error_start = forward_pass_size / 2;

  cudaHostAlloc((void**)&reg_data, reg_data_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&forward_pass, forward_pass_mem_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&forward_pass_indices, forward_pass_indices_mem_size, cudaHostAllocDefault);

  // Reserve capacity for buffers to avoid reallocations
  uint32_t num_batches = ceil(num_samples / kTrainBatchSize);
  current_buffer.outputs.reserve(num_batches);
  current_buffer.targets.reserve(num_batches);
  current_buffer.losses.reserve(num_batches);
  pending_buffer.outputs.reserve(num_batches);
  pending_buffer.targets.reserve(num_batches);
  pending_buffer.losses.reserve(num_batches);

  Logger::instance().log("registered_samples: " + std::to_string(indices.size()) + "\n");
  Logger::instance().log("Allocated registered memory for dataset: " +
                         std::to_string(reg_data_size) + " bytes\n");

  // Copy data from the original dataset to the registered memory
  std::unordered_map<size_t, size_t> index_map;
  index_map.reserve(indices.size());

  size_t i = 0;  // Counter for the registered memory
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
  if (pending_forward_pass.load()) {
    Logger::instance().log("Waiting for pending forward pass processing in destructor\n");
    forward_pass_future.wait();
  }

  cudaFreeHost(reg_data);
  cudaFreeHost(forward_pass);
  cudaFreeHost(forward_pass_indices);
  Logger::instance().log("Freed Cuda registered memory for dataset\n");
}

void RegisteredMnistTrain::processForwardPassConcurrent(ForwardPassBuffer buffer) {
  assert(buffer.outputs.size() == buffer.losses.size() && 
         "Outputs and losses vectors must have the same size");
  
  Logger::instance().log("Processing forward pass concurrently with " + 
                        std::to_string(buffer.outputs.size()) + " batches\n");

  // Process the batch results and store them in the forward_pass buffer
  size_t curr_idx = 0;
  for (size_t i = 0; i < buffer.outputs.size(); ++i) {
    processBatchResults(buffer.outputs[i], buffer.targets[i], buffer.losses[i], curr_idx);
  }
  
  Logger::instance().log("Concurrent forward pass processing completed\n");
}

/**
 * Processes batch output and stores per-sample loss and error values efficiently.
 * 
 * @param output The model's output tensor for the batch.
 * @param individual_losses The tensor containing individual losses for each sample in the batch.
 * @param curr_idx Reference to the current index in the forward_pass buffer.
 */
void RegisteredMnistTrain::processBatchResults(
    const torch::Tensor& output, 
    const torch::Tensor& targets,
    const torch::Tensor& individual_losses,
    size_t& curr_idx) {
    
  // Calculate loss for each example in the batch at once
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

void RegisteredMnistTrain::processForwardPass(const std::vector<torch::Tensor>& outputs,
                                              const std::vector<torch::Tensor>& targets,
                                              const std::vector<torch::Tensor>& losses) {
 assert(outputs.size() == losses.size() && "Outputs and losses vectors must have the same size");

  // Process the batch results and store them in the forward_pass buffer
  size_t curr_idx = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    processBatchResults(outputs[i], targets[i], losses[i], curr_idx);
  }
}

void RegisteredMnistTrain::train(size_t epoch, 
                           Net& model, 
                           torch::Device device, 
                           torch::optim::Optimizer& optimizer, 
                           size_t dataset_size) {

  Logger::instance().log("  RegTraining model for epoch " + std::to_string(epoch) + "\n");

  if (pending_forward_pass.load()) {
    Logger::instance().log("Waiting for previous forward pass processing to complete\n");
    forward_pass_future.wait();
    pending_forward_pass.store(false);
  }

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

  auto start = std::chrono::high_resolution_clock::now();

  current_buffer.outputs.clear();
  current_buffer.targets.clear(); 
  current_buffer.losses.clear();
  
  for (const auto& batch : *registered_loader) { 
    size_t batch_size = batch.size();
    
    std::vector<torch::Tensor> data_vec, target_vec;
    for (const auto& example : batch) {
      data_vec.push_back(example.data);
      target_vec.push_back(example.target);
      forward_pass_indices[global_sample_idx] = *getOriginalIndex(global_sample_idx);
      global_sample_idx++;
    }
    
    auto batch_data = torch::stack(data_vec);
    auto batch_targets = torch::stack(target_vec);
    
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
    // if (batch_idx == 0 && epoch == 1) {
    //   model.eval();
    //   torch::NoGradGuard no_grad;
    //   auto inference_output = model.forward(data);
    //   Logger::instance().log("  Going to proc batch results in batch " + std::to_string(batch_idx) + "and epoch " + std::to_string(epoch) + "\n");
    //   processBatchResults(inference_output, targets, curr_idx);
    //   model.train();  // Switch back to training mode
    // }

    // Clear gradients from previous iteration
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto individual_losses = torch::nll_loss(output, targets, {}, torch::Reduction::None);

    current_buffer.outputs.push_back(output);
    current_buffer.targets.push_back(targets);
    current_buffer.losses.push_back(individual_losses);

    // Backpropagation
    /**
     * Using sum instead of mean for the following reasons:
     * It's the most efficient (no division)
     * SGD optimizers typically handle the averaging internally
     * You still get the same training behavior
     * You keep your individual losses for analysis
     */
    auto loss_mean = individual_losses.mean();
    loss_mean.backward();
    optimizer.step();
    
    if (batch_idx % kLogInterval == 0 && worker_id % 2 == 0) {
      std::printf("\rRegTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                 epoch,
                 batch_idx * data.size(0),
                 dataset_size,
                 loss_mean.template item<float>());
    }
    
    batch_idx++;
  }

  // Wait for any remaining async operations
  if (device.is_cuda()) {
    cudaDeviceSynchronize();
  }

  Logger::instance().log("\nStarting concurrent forward pass processing for epoch " + 
                        std::to_string(epoch) + "\n");

  // Move current buffer to pending and start concurrent processing
  pending_buffer = std::move(current_buffer);
  forward_pass_future = std::async(std::launch::async, 
                                   &RegisteredMnistTrain::processForwardPassConcurrent, 
                                   this, 
                                   std::move(pending_buffer));
  pending_forward_pass.store(true);

  auto end = std::chrono::high_resolution_clock::now();
    Logger::instance().log("Total time taken server side step: " +
                          std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) +
                          " ms\n");
}

std::vector<torch::Tensor> RegisteredMnistTrain::runMnistTrain(int round, const std::vector<torch::Tensor>& w) {
  std::vector<torch::Tensor> w_cuda = updateModelParameters(w);
  size_t param_count = w_cuda.size();

  torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(GLOBAL_LEARN_RATE));

  if (round % 1 == 0) {
    std::cout << "Training model for round " << round << " epochs: " << kNumberOfEpochs << "\n";
  }

  Logger::instance().log("Training model for step " + std::to_string(round) + " epochs: " + std::to_string(kNumberOfEpochs) + "\n");

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, optimizer, subset_size);
  }

  // if (round % 2 == 0) {
  //   Logger::instance().log("Testing model after training round " + std::to_string(round) + "\n");
  //   test(model, device, *test_loader, test_dataset_size);
  // }

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
  // Wait for any pending forward pass processing
  if (pending_forward_pass.load()) {
    Logger::instance().log("Waiting for pending forward pass processing before inference\n");
    forward_pass_future.wait();
    pending_forward_pass.store(false);
  }

  torch::NoGradGuard no_grad;  // Prevent gradient calculation
  model.eval();                // Set model to evaluation mode
  updateModelParameters(w);
  
  Logger::instance().log("  Running inference on registered dataset\n");

  int32_t correct = 0;
  size_t total = 0;
  
  // For tracking loss and error rate indices in the forward pass buffer
  size_t curr_idx = 0;
  size_t global_sample_idx = 0;  // Track global sample position
  
  std::vector<torch::Tensor> outputs_vec;
  std::vector<torch::Tensor> targets_vec;
  std::vector<torch::Tensor> losses_vec;

  uint32_t num_batches = ceil(num_samples / kTrainBatchSize);
  outputs_vec.reserve(num_batches);
  targets_vec.reserve(num_batches);
  losses_vec.reserve(num_batches);

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
    auto individual_losses = torch::nll_loss(output, targets, {}, torch::Reduction::None);

    outputs_vec.push_back(output);
    targets_vec.push_back(targets);
    losses_vec.push_back(individual_losses);
  }

  // Wait for any remaining async operations
  if (device.is_cuda()) {
    cudaDeviceSynchronize();
  }

  // For inference, process synchronously since we need immediate results
  Logger::instance().log("Processing inference results\n");
  processForwardPass(outputs_vec, targets_vec, losses_vec);
  
  Logger::instance().log("Inference completed - Loss: " + std::to_string(loss) + 
                        ", Error rate: " + std::to_string(error_rate) + 
                        ", Total samples: " + std::to_string(total) + "\n");
}