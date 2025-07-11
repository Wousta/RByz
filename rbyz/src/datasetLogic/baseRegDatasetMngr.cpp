#include "datasetLogic/baseRegDatasetMngr.hpp"
#include "global/globalConstants.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"
#include <cstddef>
#include <random>
#include <torch/types.h>
#include <vector>

template <typename NetType>
BaseRegDatasetMngr<NetType>::BaseRegDatasetMngr(int worker_id, TrainInputParams &t_params,
                                                NetType net)
    : IRegDatasetMngr(worker_id, t_params),
      model(std::move(net)), device(init_device()) {
  torch::manual_seed(1);
  model->to(device);

  if (device.is_cuda()) {
    cudaStreamCreate(&memcpy_stream_A);
    cudaStreamCreate(&memcpy_stream_B);
  }
}

template <typename NetType> BaseRegDatasetMngr<NetType>::~BaseRegDatasetMngr() {
  if (pending_forward_pass.load()) {
    Logger::instance().log(
        "Waiting for pending forward pass processing in destructor\n");
    forward_pass_future.wait();
  }

  if (device.is_cuda()) {
    cudaError_t err_A, err_B;
    err_A = cudaStreamDestroy(memcpy_stream_A);
    err_B = cudaStreamDestroy(memcpy_stream_B);
    if (err_A != cudaSuccess) {
      std::cerr << "Error destroying stream A: " << cudaGetErrorString(err_A)
                << std::endl;
    }
    if (err_B != cudaSuccess) {
      std::cerr << "Error destroying stream B: " << cudaGetErrorString(err_B)
                << std::endl;
    }

    cudaFreeHost(data_info.reg_data);

    if (!only_flt) {
      cudaFreeHost(f_pass_data.forward_pass);
      cudaFreeHost(f_pass_data.forward_pass_indices);
    }
    Logger::instance().log("Freed Cuda registered memory for dataset\n");
  }
}

template <typename NetType>
torch::Device BaseRegDatasetMngr<NetType>::init_device() {
  try {
    if (torch::cuda::is_available()) {
      Logger::instance().log("CUDA is available, using GPU\n");
      return torch::Device(torch::kCUDA);
    } else {
      Logger::instance().log("CUDA is not available, using CPU\n");
      return torch::Device(torch::kCPU);
    }
  } catch (const c10::Error &e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    std::cout << "Falling back to CPU" << std::endl;
    return torch::Device(torch::kCPU);
  }
}

template <typename NetType>
std::vector<torch::Tensor> BaseRegDatasetMngr<NetType>::getInitialWeights() {
  std::vector<torch::Tensor> initialWeights;
  // Must be at CPU at the beginning
  for (const auto &param : model->parameters()) {
    initialWeights.push_back(param.clone().detach().to(torch::kCPU));
  }

  for (const auto &buffer : model->buffers()) {
    initialWeights.push_back(buffer.clone().detach().to(torch::kCPU));
  }

  return initialWeights;
}

template <typename NetType>
std::vector<torch::Tensor> BaseRegDatasetMngr<NetType>::calculateUpdateCuda(
    const std::vector<torch::Tensor> &w_cuda) {
  std::vector<torch::Tensor> params = model->parameters();
  std::vector<torch::Tensor> buffers = model->buffers();
  std::vector<torch::Tensor> result;
  result.reserve(w_cuda.size());

  for (size_t i = 0; i < params.size(); i++) {
    // Subtract on GPU
    auto update = params[i].clone().detach() - w_cuda[i];

    // Copy result to CPU
    auto cpu_update = torch::empty_like(update, torch::kCPU);
    cudaMemcpyAsync(cpu_update.data_ptr<float>(), update.data_ptr<float>(),
                    update.numel() * sizeof(float), cudaMemcpyDeviceToHost,
                    memcpy_stream_A);
    result.push_back(cpu_update);
  }

  for (size_t i = 0; i < buffers.size(); i++) {
    // Subtract on GPU
    auto update = buffers[i].clone().detach() - w_cuda[params.size() + i];
    auto cpu_update = torch::empty_like(update, torch::kCPU);

    // Copy result to CPU
    if (update.dtype() == torch::kFloat32) {
      cudaMemcpyAsync(cpu_update.data_ptr<float>(), update.data_ptr<float>(),
                      update.numel() * sizeof(float), cudaMemcpyDeviceToHost,
                      memcpy_stream_A);
    } else if (update.dtype() == torch::kInt64) {
      cudaMemcpyAsync(cpu_update.data_ptr<int64_t>(), update.data_ptr<int64_t>(),
                      update.numel() * sizeof(int64_t), cudaMemcpyDeviceToHost,
                      memcpy_stream_B);
    } else {
      cpu_update.copy_(update);
    }
    result.push_back(cpu_update);
  }

  cudaDeviceSynchronize();
  return result;
}

template <typename NetType>
std::vector<torch::Tensor> BaseRegDatasetMngr<NetType>::calculateUpdateCPU(
    const std::vector<torch::Tensor> &w) {
  std::vector<torch::Tensor> params = model->parameters();
  std::vector<torch::Tensor> buffers = model->buffers();
  std::vector<torch::Tensor> result;
  result.reserve(w.size());
  
  for (size_t i = 0; i < params.size(); i++) {
    // Subtract on CPU
    auto update = params[i].clone().detach() - w[i];
    result.push_back(update);
  }

  for (size_t i = 0; i < buffers.size(); i++) {
    // Subtract on CPU
    auto update = buffers[i].clone().detach() - w[params.size() + i];
    result.push_back(update);
  }

  return result;
}

template <typename NetType>
template <typename DataLoader>
void BaseRegDatasetMngr<NetType>::test(DataLoader &data_loader) {
  torch::NoGradGuard no_grad;
  model->eval();
  double test_loss = 0;
  int correct = 0;
  int total_samples = 0; // Track actual samples processed
  int TPi = 0; // True Positives for source class
  int FNi = 0; // False Negatives for source class
  missclassed_samples = 0;

  int batch_idx = 0;

  Logger::instance().log("  Testing:\n");
  auto params = model->parameters();
  auto buffers = model->buffers();

  for (const auto &batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model->forward(data);
    // test_loss += torch::nll_loss(output, targets,
    //                              /*weight=*/{}, torch::Reduction::Sum)
    //                  .template item<float>();
    test_loss += torch::nn::functional::cross_entropy(output, targets, 
    torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum)).template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
    total_samples += targets.size(0); // Add actual batch size

    if (src_class != INACTIVE) {
      // Calculate source class recall
      for (int i = 0; i < targets.size(0); ++i) {
        int64_t true_label = targets[i].template item<int64_t>();
        int64_t predicted_label = pred[i].template item<int64_t>();

        if (true_label == src_class && predicted_label == target_class) {
          missclassed_samples++;
        }

        if (true_label == src_class) {
          if (predicted_label == src_class) {
            TPi++;
          } else {
            FNi++;
          }
        }
      }
    }
  }

  test_loss /= total_samples; 
  this->test_loss = test_loss; 
  this->test_accuracy =
      static_cast<float>(correct) * 100 / total_samples; 

  

  std::ostringstream oss;
  oss << "\n  Test set: Average loss: " << std::fixed << std::setprecision(4)
      << test_loss << " | Accuracy: " << std::fixed << std::setprecision(3)
      << test_accuracy << " (" << correct << "/" << total_samples << ")";
      
  if (src_class != INACTIVE) {
    src_class_recall = static_cast<float>(TPi * 100) / (TPi + FNi);
    oss << " | Source class recall: " << std::fixed << std::setprecision(3)
        << src_class_recall << " (TP: " << TPi << ", FN: " << FNi << ")";
  }

  Logger::instance().log(oss.str() + "\n");
  Logger::instance().log("  Testing done\n");
}

template <typename NetType>
template <typename DataLoader>
void BaseRegDatasetMngr<NetType>::train(size_t epoch,
                                        torch::optim::Optimizer &optimizer,
                                        DataLoader &data_loader) {
  Logger::instance().log("  RegTraining model for epoch " +
                         std::to_string(epoch) + "\n");

  if (!only_flt) {
    if (pending_forward_pass.load()) {
      Logger::instance().log(
          "Waiting for previous forward pass processing to complete\n");
      forward_pass_future.wait();
      pending_forward_pass.store(false);
    }

    current_buffer.outputs.clear();
    current_buffer.targets.clear();
    current_buffer.losses.clear();
  }                       

  model->train();

  size_t batch_idx = 0;
  size_t global_sample_idx = 0; // Track global sample position

  // For tracking loss and error rate indices in the forward pass buffer
  size_t curr_idx = 0;

  // Track total loss for averaging
  double total_loss = 0.0;
  size_t total_samples = 0;
  size_t total = 0;
  int32_t correct = 0;

  std::vector<int> labels(10, 0);
  for (const auto &batch : data_loader) {
    if (!only_flt) {
      for (int i = 0; i < batch.data.size(0); ++i) {
        labels[batch.target[i].template item<int64_t>()]++;
        f_pass_data.forward_pass_indices[global_sample_idx] = *getOriginalIndex(global_sample_idx);
        global_sample_idx++;
      }
    }

    auto data_cpu = batch.data;
    auto targets_cpu = batch.target;
    torch::Tensor data, targets;
    if (device.is_cuda()) {
      // Create CUDA tensors with the same shape and type
      data = torch::empty_like(data_cpu, torch::TensorOptions().device(device));
      targets =
          torch::empty_like(targets_cpu, torch::TensorOptions().device(device));

      // Asynchronously copy data from CPU to GPU
      cudaMemcpyAsync(data.data_ptr<float>(), data_cpu.template data_ptr<float>(),
                      data_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice,
                      memcpy_stream_A);

      cudaMemcpyAsync(targets.data_ptr<int64_t>(),
                      targets_cpu.template data_ptr<int64_t>(),
                      targets_cpu.numel() * sizeof(int64_t),
                      cudaMemcpyHostToDevice, memcpy_stream_B);

      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    } else {
      // For CPU, just use the original tensors
      data = data_cpu;
      targets = targets_cpu;
    }

    // Switch to evaluation mode for writing forward pass results
    // if (batch_idx == 0 && epoch == 1) {
    //   model.eval();
    //   torch::NoGradGuard no_grad;
    //   auto inference_output = model.forward(data);
    //   Logger::instance().log("  Going to proc batch results in batch " +
    //   std::to_string(batch_idx) + "and epoch " + std::to_string(epoch) +
    //   "\n"); processBatchResults(inference_output, targets, curr_idx);
    //   model.train();  // Switch back to training mode
    // }

    // Clear gradients from previous iteration
    optimizer.zero_grad();
    auto output = model->forward(data);

    if (!only_flt) {
      auto individual_losses = torch::nn::functional::cross_entropy(output, targets, 
      torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
      // auto individual_losses =
      //     torch::nll_loss(output, targets, {}, torch::Reduction::None);

      current_buffer.outputs.push_back(output);
      current_buffer.targets.push_back(targets);
      current_buffer.losses.push_back(individual_losses);
    }
    // Backpropagation
    /**
     * Using sum instead of mean for the following reasons:
     * It's the most efficient (no division)
     * SGD optimizers typically handle the averaging internally
     * You still get the same training behavior
     * You keep your individual losses for analysis
     */
    //auto loss_mean = individual_losses.mean();
    auto loss_mean = torch::nn::functional::cross_entropy(output, targets);
    AT_ASSERT(!std::isnan(loss_mean.template item<float>()));
    loss_mean.backward();
    optimizer.step();

    // if (batch_idx % kLogInterval == 0 && worker_id % 5 == 0) {
    //   std::printf("\rRegTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f", epoch,
    //               batch_idx * data.size(0), subset_size,
    //               loss_mean.template item<float>());
    // }

    batch_idx++;
  }

  // Logger::instance().log("Labels per class processed: " +
  //                        std::to_string(labels[0]) + ", " +
  //                        std::to_string(labels[1]) + ", " +
  //                        std::to_string(labels[2]) + ", " +
  //                        std::to_string(labels[3]) + ", " +
  //                        std::to_string(labels[4]) + ", " +
  //                        std::to_string(labels[5]) + ", " +
  //                        std::to_string(labels[6]) + ", " +
  //                        std::to_string(labels[7]) + ", " +
  //                        std::to_string(labels[8]) + ", " +
  //                        std::to_string(labels[9]) + "\n");

  // Wait for any remaining async operations
  if (device.is_cuda()) {
    cudaDeviceSynchronize();
  }

  if (!only_flt) {
    Logger::instance().log(
        "\nStarting concurrent forward pass processing for epoch " +
        std::to_string(epoch) + "\n");

    // Move current buffer to pending and start concurrent processing
    pending_buffer = std::move(current_buffer);
    forward_pass_future = std::async(
        std::launch::async, &BaseRegDatasetMngr::processForwardPassConcurrent,
        this, std::move(pending_buffer));
    pending_forward_pass.store(true);
  }
}

template <typename NetType>
std::vector<torch::Tensor> BaseRegDatasetMngr<NetType>::updateModelParameters(
    const std::vector<torch::Tensor> &w) {
  std::vector<torch::Tensor> params = model->parameters();
  std::vector<torch::Tensor> buffers = model->buffers();
  size_t param_count = params.size();
  size_t buffer_count = buffers.size();
  std::vector<torch::Tensor> w_cuda;

  if (device.is_cuda()) {
    // Copy parameters and buffers to CUDA tensors first
    w_cuda.reserve(param_count + buffer_count);
    for (const auto &param : w) {
      auto cuda_tensor = torch::empty_like(param, torch::kCUDA);

      if (param.dtype() == torch::kFloat32) {
        cudaMemcpyAsync(cuda_tensor.data_ptr<float>(), param.data_ptr<float>(),
                              param.numel() * sizeof(float), cudaMemcpyHostToDevice,
                              memcpy_stream_A);
      } else if (param.dtype() == torch::kInt64) {
        cudaMemcpyAsync(cuda_tensor.data_ptr<int64_t>(), param.data_ptr<int64_t>(),
                              param.numel() * sizeof(int64_t), cudaMemcpyHostToDevice,
                              memcpy_stream_B);
      } else {
        // For other types, use PyTorch's built-in copy (slower but safe)
        cuda_tensor.copy_(param);
      }
      
      w_cuda.push_back(cuda_tensor);
    }

    cudaDeviceSynchronize();

    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w_cuda[i]);
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers[i].data().copy_(w_cuda[param_count + i]);
    }
    return w_cuda;
  } else {
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w[i]);
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers[i].data().copy_(w[param_count + i]);
    }
    return w;
  }
}

template <typename NetType>
template <typename DataLoader>
void BaseRegDatasetMngr<NetType>::runInferenceBase(DataLoader &data_loader) {
  if (only_flt) {
    Logger::instance().log("WARNING: Skipping inference in FLT mode\n");
    return;
  }

  // Wait for any pending forward pass processing
  if (pending_forward_pass.load()) {
    Logger::instance().log(
        "Waiting for pending forward pass processing before inference\n");
    forward_pass_future.wait();
    pending_forward_pass.store(false);
  }

  torch::NoGradGuard no_grad; // Prevent gradient calculation
  model->eval();              // Set model to evaluation mode

  int32_t correct = 0;
  size_t total = 0;

  // For tracking loss and error rate indices in the forward pass buffer
  size_t curr_idx = 0;
  size_t global_sample_idx = 0; // Track global sample position

  inference_buffer.outputs.clear();
  inference_buffer.targets.clear();
  inference_buffer.losses.clear();

  for (const auto &batch : data_loader) {
    for (int i = 0; i < batch.data.size(0); ++i) {
      f_pass_data.forward_pass_indices[global_sample_idx] =
          *getOriginalIndex(global_sample_idx);
      global_sample_idx++;
    }

    auto data_cpu = batch.data;
    auto targets_cpu = batch.target;
    torch::Tensor data, targets;
    if (device.is_cuda()) {
      // Create CUDA tensors with the same shape and type
      data = torch::empty_like(data_cpu, torch::TensorOptions().device(device));
      targets =
          torch::empty_like(targets_cpu, torch::TensorOptions().device(device));

      // Asynchronously copy data from CPU to GPU
      cudaMemcpyAsync(data.data_ptr<float>(), data_cpu.template data_ptr<float>(),
                      data_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice,
                      memcpy_stream_A);

      cudaMemcpyAsync(targets.data_ptr<int64_t>(),
                      targets_cpu.template data_ptr<int64_t>(),
                      targets_cpu.numel() * sizeof(int64_t),
                      cudaMemcpyHostToDevice, memcpy_stream_B);

      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    } else {
      // For CPU, just use the original tensors
      data = data_cpu;
      targets = targets_cpu;
    }

    // Run forward pass
    auto output = model->forward(data);
    // auto individual_losses =
    //     torch::nll_loss(output, targets, {}, torch::Reduction::None);
    auto individual_losses = torch::nn::functional::cross_entropy(
        output, targets, 
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)
    );

    inference_buffer.outputs.push_back(output);
    inference_buffer.targets.push_back(targets);
    inference_buffer.losses.push_back(individual_losses);
  }

  // Wait for any remaining async operations
  if (device.is_cuda()) {
    cudaDeviceSynchronize();
  }

  // For inference, process synchronously since we need immediate results
  processForwardPass(inference_buffer);

  Logger::instance().log("Inference completed - Loss: " + std::to_string(loss) +
                         ", Error rate: " + std::to_string(error_rate) +
                         ", Total samples: " + std::to_string(total) + "\n");
}

template <typename NetType>
std::vector<size_t>
BaseRegDatasetMngr<NetType>::getClientsSamplesCount(uint32_t clnt_subset_size,
                                                    uint32_t srvr_subset_size,
                                                    uint32_t dataset_size) {
  Logger::instance().log("Getting samples count for each client\n");
  std::vector<size_t> samples_count(n_clients, 0);

  // Allocate indices to workers
  float srvr_proportion = static_cast<float>(srvr_subset_size) / dataset_size;
  float clnt_proportion = (1 - srvr_proportion) / (n_clients);
  for (size_t i = 0; i < n_clients; i++) {
    std::vector<size_t> worker_indices;
    for (const auto &[label, indices] : label_to_indices) {
      size_t total_samples = indices.size();
      size_t samples_clnt =
          static_cast<size_t>(std::floor(total_samples * clnt_proportion));
      size_t samples_srvr =
          static_cast<size_t>(std::ceil(total_samples * srvr_proportion));

      size_t start =
          static_cast<size_t>(std::ceil(samples_srvr + i * samples_clnt));
      size_t end;
      if (i == n_clients - 1) {
        end = static_cast<size_t>(
            std::ceil(samples_srvr + (i + 1) * samples_clnt));
        end = total_samples; // Last worker gets the remaining samples
      } else {
        end = start + samples_clnt;
      }

      // Add the worker's portion of indices for this label
      worker_indices.insert(worker_indices.end(), indices.begin() + start,
                            indices.begin() + end);
    }

    if (worker_indices.size() > clnt_subset_size) {
      worker_indices.resize(clnt_subset_size);
    }

    size_t worker_samples_count = worker_indices.size();
    samples_count[i] = worker_samples_count;
  }

  return samples_count;
}

template <typename NetType>
SubsetSampler BaseRegDatasetMngr<NetType>::get_subset_sampler(
    int worker_id_arg, size_t dataset_size_arg, int64_t subset_size_arg, uint32_t srvr_subset_size,
    const std::map<int64_t, std::vector<size_t>> &label_to_indices) {

  // These first two ifs were for debugging purposes
  if (subset_size_arg == dataset_size_arg) {
    std::vector<size_t> worker_indices(dataset_size_arg);
    std::iota(worker_indices.begin(), worker_indices.end(), 0);
    return SubsetSampler(worker_indices);
  }

  // Allocate indices to workers
  float srvr_proportion =
      static_cast<float>(srvr_subset_size) / dataset_size_arg;
  float clnt_proportion = (1 - srvr_proportion) / (n_clients);

  std::vector<size_t> worker_indices;
  for (const auto &[label, indices] : label_to_indices) {
    size_t total_samples = indices.size();
    
    size_t samples_srvr =
        static_cast<size_t>(std::ceil(total_samples * srvr_proportion));
    size_t samples_clnt =
        static_cast<size_t>(std::floor(total_samples * clnt_proportion));

    size_t start;
    size_t end;
    if (worker_id_arg == 0) {
      start = 0; // Server gets the first portion
      end = samples_srvr;
    } else {
      start = static_cast<size_t>(
          std::ceil(samples_srvr + (worker_id_arg - 1) * samples_clnt));
      if (worker_id_arg == n_clients) {
        end = total_samples; // Last worker gets the remaining samples
      } else {
        end = start + samples_clnt;
      }
    }

    // Add the worker's portion of indices for this label
    worker_indices.insert(worker_indices.end(), indices.begin() + start,
                          indices.begin() + end);
  }

  std::random_device rd;
  std::mt19937 rng(rd());
  std::shuffle(worker_indices.begin(), worker_indices.end(), rng);

  // If the subset size is smaller than the allocated indices, truncate
  if (worker_indices.size() > subset_size_arg) {
    worker_indices.resize(subset_size_arg);
  }

  Logger::instance().log("Worker " + std::to_string(worker_id) +
                         " using stratified indices of size: " +
                         std::to_string(worker_indices.size()) + "\n");

  return SubsetSampler(worker_indices);
}

template <typename NetType>
void BaseRegDatasetMngr<NetType>::processForwardPassConcurrent(
    ForwardPassBuffer buffer) {
  assert(buffer.outputs.size() == buffer.losses.size() &&
         "Outputs and losses vectors must have the same size");

  // Process the batch results and store them in the forward_pass buffer
  size_t curr_idx = 0;
  for (size_t i = 0; i < buffer.outputs.size(); ++i) {
    processBatchResults(buffer.outputs[i], buffer.targets[i], buffer.losses[i],
                        curr_idx);
  }
}

/**
 * Processes batch output and stores per-sample loss and error values
 * efficiently.
 *
 * @param output The model's output tensor for the batch.
 * @param individual_losses The tensor containing individual losses for each
 * sample in the batch.
 * @param curr_idx Reference to the current index in the forward_pass buffer.
 */
template <typename NetType>
void BaseRegDatasetMngr<NetType>::processBatchResults(
    const torch::Tensor &output, const torch::Tensor &targets,
    const torch::Tensor &individual_losses, size_t &curr_idx) {

  // Calculate loss for each example in the batch at once
  auto predictions = output.argmax(1);
  auto correct_predictions = predictions.eq(targets);

  // Copy results to CPU if on GPU
  torch::Tensor cpu_losses, cpu_correct;
  if (device.is_cuda()) {
    cpu_losses = torch::empty_like(individual_losses, torch::kCPU);
    cpu_correct = torch::empty_like(correct_predictions, torch::kCPU);

    cudaMemcpyAsync(cpu_losses.data_ptr<float>(),
                    individual_losses.data_ptr<float>(),
                    individual_losses.numel() * sizeof(float),
                    cudaMemcpyDeviceToHost, memcpy_stream_A);

    cudaMemcpyAsync(cpu_correct.data_ptr<bool>(),
                    correct_predictions.data_ptr<bool>(),
                    correct_predictions.numel() * sizeof(bool),
                    cudaMemcpyDeviceToHost, memcpy_stream_B);

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
      f_pass_data.forward_pass[curr_idx] = losses_accessor[i];
      f_pass_data.forward_pass[error_start + curr_idx] = correct_accessor[i] ? 0.0f : 1.0f;
      if (curr_idx == 0)
        Logger::instance().log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      if (curr_idx < 3) {
        Logger::instance().log(
            "Processed sample " + std::to_string(curr_idx) +
            " with original index: " +
            std::to_string(f_pass_data.forward_pass_indices[curr_idx]) +
            " label: " + std::to_string(*getLabel(curr_idx)) + " with loss: " +
            std::to_string(f_pass_data.forward_pass[curr_idx]) +
            " and error: " +
            std::to_string(f_pass_data.forward_pass[error_start + curr_idx]) +
            "\n");
      }

      curr_idx++;
    } else {
      throw std::runtime_error("Forward pass buffer overflow"
                               " - current index: " +
                               std::to_string(curr_idx) + ", max size: " +
                               std::to_string(forward_pass_size / 2));
    }
  }
}

template <typename NetType>
void BaseRegDatasetMngr<NetType>::processForwardPass(ForwardPassBuffer buffer) {
  assert(buffer.outputs.size() == buffer.losses.size() &&
         "Outputs and losses vectors must have the same size");

  // Process the batch results and store them in the forward_pass buffer
  size_t curr_idx = 0;
  for (size_t i = 0; i < buffer.outputs.size(); ++i) {
    processBatchResults(buffer.outputs[i], buffer.targets[i], buffer.losses[i],
                        curr_idx);
  }
}

template <typename NetType>
void BaseRegDatasetMngr<NetType>::initDataInfo(
    const std::vector<size_t> &indices, int img_size) {
  data_info.num_samples = indices.size();
  // data_info.image_size = img_size * sizeof(uint8_t); // UINT8CHANGE
  data_info.image_size = img_size * sizeof(float);
  Logger::instance().log(
      "Initializing data info with " + std::to_string(data_info.num_samples) +
      " samples and image size: " + std::to_string(data_info.image_size) +
      " bytes\n");
  data_info.reg_data_size = indices.size() * data_info.get_sample_size();
  f_pass_data.forward_pass_mem_size = data_info.num_samples *
                                      f_pass_data.values_per_sample *
                                      f_pass_data.bytes_per_value;
  f_pass_data.forward_pass_indices_mem_size =
      data_info.num_samples * sizeof(uint32_t);

  forward_pass_size =
      f_pass_data.forward_pass_mem_size / f_pass_data.bytes_per_value;
  error_start = forward_pass_size / 2;

  cudaHostAlloc((void **)&data_info.reg_data, data_info.reg_data_size,
                cudaHostAllocDefault);

  if (!only_flt) {
    // These memory allocations are not needed in FL trust
    cudaHostAlloc((void **)&f_pass_data.forward_pass,
                  f_pass_data.forward_pass_mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&f_pass_data.forward_pass_indices,
                  f_pass_data.forward_pass_indices_mem_size,
                  cudaHostAllocDefault);

    uint32_t num_batches = ceil(data_info.num_samples / kTrainBatchSize);
    current_buffer.outputs.reserve(num_batches);
    current_buffer.targets.reserve(num_batches);
    current_buffer.losses.reserve(num_batches);
    pending_buffer.outputs.reserve(num_batches);
    pending_buffer.targets.reserve(num_batches);
    pending_buffer.losses.reserve(num_batches);
    inference_buffer.outputs.reserve(num_batches);
    inference_buffer.targets.reserve(num_batches);
    inference_buffer.losses.reserve(num_batches);
  }

  Logger::instance().log(
      "registered_samples: " + std::to_string(indices.size()) + "\n");
  Logger::instance().log("Allocated registered memory for dataset: " +
                         std::to_string(data_info.reg_data_size) + " bytes\n");
}

//////////////////////// LABEL FLIPPING ATTCKS ////////////////////////
template <typename NetType>
void BaseRegDatasetMngr<NetType>::flipLabelsRandom(float flip_ratio,
                                                   std::mt19937 &rng) {
  if (flip_ratio <= 0.0f || flip_ratio >= 1.0f) {
    throw std::invalid_argument("Flip ratio must be between 0 and 1");
  }

  size_t num_to_flip = static_cast<size_t>(data_info.num_samples * flip_ratio);
  std::uniform_int_distribution<size_t> sample_dist(0, data_info.num_samples - 1);
  std::uniform_int_distribution<int> label_dist(0, 9); // MNIST has 10 classes

  std::unordered_set<size_t> flipped_indices;

  // If overwrite_poisoned is false, keep the VD section of the dataset intact (first indices)
  int safe_labels = 0;
  if (!overwrite_poisoned) {
    safe_labels = std::ceil(data_info.num_samples * t_params.clnt_vd_proportion);
    for (size_t i = 0; i < safe_labels; ++i) {
      flipped_indices.insert(i);
    }
  }

  Logger::instance().log(
      "Starting random label flipping attack: " + std::to_string(num_to_flip) +
      " samples (" + std::to_string(flip_ratio * 100) + "%)\n");

  while (flipped_indices.size() < num_to_flip + safe_labels) {
    size_t idx = sample_dist(rng);
    if (flipped_indices.find(idx) == flipped_indices.end()) {
      int64_t *label_ptr = getLabel(idx);
      int64_t original_label = *label_ptr;

      // Generate a different label
      int64_t new_label;
      do {
        new_label = label_dist(rng);
      } while (new_label == original_label);

      *label_ptr = new_label;
      flipped_indices.insert(idx);
    }
  }

  Logger::instance().log("Random label flipping completed\n");
}

template <typename NetType>
void BaseRegDatasetMngr<NetType>::flipLabelsTargeted(int source_label,
                                                     int target_label,
                                                     float flip_ratio,
                                                     std::mt19937 &rng) {
  if (source_label < 0 || source_label > 9 || target_label < 0 ||
      target_label > 9) {
    throw std::invalid_argument("Labels must be between 0 and 9 for MNIST");
  }

  if (source_label == target_label) {
    throw std::invalid_argument("Source and target labels must be different");
  }

  // Find all samples with the source label
  std::vector<size_t> source_indices = findSamplesWithLabel(source_label);

  if (source_indices.empty()) {
    Logger::instance().log("No samples found with source label " +
                           std::to_string(source_label) + "\n");
    return;
  }

  size_t num_to_flip = static_cast<size_t>(source_indices.size() * flip_ratio);

  Logger::instance().log("Starting targeted label flipping attack: " +
                         std::to_string(source_label) + " -> " +
                         std::to_string(target_label) + " (" +
                         std::to_string(num_to_flip) + " samples)\n");

  // Randomly select which source samples to flip
  std::shuffle(source_indices.begin(), source_indices.end(), rng);

  for (size_t i = 0; i < num_to_flip && i < source_indices.size(); ++i) {
    size_t idx = source_indices[i];
    *getLabel(idx) = target_label;
  }

  Logger::instance().log("Targeted label flipping completed\n");
}

template <typename NetType>
void BaseRegDatasetMngr<NetType>::corruptImagesRandom(float flip_ratio,
                                                      std::mt19937 &rng) {
  if (flip_ratio <= 0.0f || flip_ratio >= 1.0f) {
    throw std::invalid_argument("Flip ratio must be between 0 and 1");
  }

  size_t num_to_corrupt = static_cast<size_t>(data_info.num_samples * flip_ratio);
  std::uniform_int_distribution<size_t> sample_dist(0, data_info.num_samples - 1);

  std::unordered_set<size_t> corrupted_indices;

  // If overwrite_poisoned is false, keep the VD section of the dataset intact (first indices)
  int safe_labels = 0;
  if (!overwrite_poisoned) {
    safe_labels = std::ceil(data_info.num_samples * t_params.clnt_vd_proportion);
    for (size_t i = 0; i < safe_labels; ++i) {
      corrupted_indices.insert(i);
    }
  }

  if (num_to_corrupt > data_info.num_samples - corrupted_indices.size()) {
    num_to_corrupt = data_info.num_samples - corrupted_indices.size();
  }

  Logger::instance().log(
      "Starting random image corruption attack: " + std::to_string(num_to_corrupt) +
      " samples (" + std::to_string(flip_ratio * 100) + "%)\n");

  while (corrupted_indices.size() < num_to_corrupt + safe_labels) {
    size_t idx = sample_dist(rng);
    if (corrupted_indices.find(idx) == corrupted_indices.end()) {
      float *image_ptr = getImage(idx);
      
      // Set all image pixels to zero (garbage values)
      size_t num_pixels = data_info.image_size / sizeof(float);
      std::memset(image_ptr, 0, data_info.image_size);
      
      corrupted_indices.insert(idx);
    }
  }

  Logger::instance().log("Random image corruption completed\n");
}

template <typename NetType>
std::vector<size_t>
BaseRegDatasetMngr<NetType>::findSamplesWithLabel(int label) {
  std::vector<size_t> indices;
  indices.reserve(data_info.num_samples / 10); // Rough estimate for MNIST and CIFAR-10

  for (size_t i = 0; i < data_info.num_samples; ++i) {
    if (*getLabel(i) == label) {
      indices.push_back(i);
    }
  }

  return indices;
}

// Explicit template instantiation for the types we need
#include "nets/cifar10Net.hpp"
#include "nets/mnistNet.hpp"

template class BaseRegDatasetMngr<MnistNet>;
template class BaseRegDatasetMngr<Cifar10Net>;