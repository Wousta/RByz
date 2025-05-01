#include "../include/mnistTrain.hpp"
#include "../include/logger.hpp"
#include "../include/tensorOps.hpp"


#include <cstddef>
#include <cstdio>
#include <iostream> 
#include <string>
#include <vector>
#include <cuda_runtime.h>

const char* kDataRoot = "./data";
auto train_dataset_default = torch::data::datasets::MNIST(kDataRoot)
  .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
  .map(torch::data::transforms::Stack<>());
const size_t train_dataset_size_default = train_dataset_default.size().value();
auto train_loader_default = 
  torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(train_dataset_default), 64);

MnistTrain::MnistTrain(int worker_id, int64_t subset_size) 
  : device(init_device()),
    subset_size(subset_size),
    worker_id(worker_id),
    train_dataset(torch::data::datasets::MNIST(kDataRoot)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>())),
    test_dataset(torch::data::datasets::MNIST(
      kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>()))
{
    // Worker with id = 0 is the server, needs different random seed for the sampler randperm
    if (worker_id == 0) {
        torch::manual_seed(1);
    } else {
        torch::manual_seed(2);
    }
    
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

SubsetSampler MnistTrain::get_subset_sampler(int worker_id, size_t dataset_size, int64_t subset_size) {
  // Create sequential indices instead of random permutation
  std::vector<int64_t> sequential_indices(dataset_size);
  std::iota(sequential_indices.begin(), sequential_indices.end(), 0); 
  auto indices_tensor = torch::tensor(sequential_indices);

  int64_t start = worker_id * subset_size;
  if(worker_id == 0) {
    start = 0;
  }
  
  int64_t end = start + subset_size;
  if (end > dataset_size) {
    end = dataset_size;
  }

  auto subset_tensor = indices_tensor.slice(0, start, end);

  std::vector<size_t> subset_indices(subset_size);
  for (int64_t i = 0; i < subset_size && i < (end - start); ++i) {
    subset_indices[i] = static_cast<size_t>(subset_tensor[i].item<int64_t>());
  }
  
  // If the subset size is larger than available data, resize the vector
  if (subset_size > (end - start)) {
    subset_indices.resize(end - start);
  }

  Logger::instance().log("Worker " + std::to_string(worker_id) + 
                        " using sequential indices: " + std::to_string(start) + 
                        " to " + std::to_string(end-1) + "\n");

  return SubsetSampler(subset_indices);
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

std::vector<torch::Tensor> MnistTrain::runMnistTrainDummy(std::vector<torch::Tensor>& w) {
  std::cout << "Running dummy MNIST training\n";
  
  for(size_t i = 0; i < w.size(); i++) {
    w[i] = w[i] + 1;  // element-wise addition of 1
  }
  
  return w;
}

template <typename DataLoader>
void MnistTrain::train(
    size_t epoch,
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
      cudaMemcpyAsync(
        data_device.data_ptr<float>(),
        batch.data.template data_ptr<float>(),
        batch.data.numel() * sizeof(float),
        cudaMemcpyHostToDevice
      );
      
      cudaMemcpyAsync(
        targets_device.data_ptr<int64_t>(),
        batch.target.template data_ptr<int64_t>(),
        batch.target.numel() * sizeof(int64_t),
        cudaMemcpyHostToDevice
      );
      
      // Ensure copy is complete before proceeding
      cudaDeviceSynchronize();
    } else {
      // For CPU, just use the original tensors
      data_device = batch.data;
      targets_device = batch.target;
    }

    {
      std::ostringstream oss;
      oss << "  Targets (first 10 elements): [";
      int num_to_print = std::min(10, static_cast<int>(targets_device.numel()));
      for (int i = 0; i < num_to_print; ++i) {
        oss << targets_device[i].item<int64_t>();
        if (i < num_to_print - 1) oss << ", ";
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

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          nll_loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void MnistTrain::test(
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
  std::ostringstream oss;
  oss << "\nTest set: Average loss: " << std::fixed << std::setprecision(4) << test_loss
      << " | Accuracy: " << std::fixed << std::setprecision(3) 
      << static_cast<double>(correct) / dataset_size << std::endl;
  Logger::instance().log(oss.str());
  Logger::instance().log("Testing done\n");

  
}

std::vector<torch::Tensor> MnistTrain::runMnistTrain(int round, const std::vector<torch::Tensor>& w) {
  // Update model parameters, w is in cpu so if device is cuda copy to device is needed
  std::vector<torch::Tensor> params = model.parameters();
  size_t param_count = model.parameters().size();
  std::vector<torch::Tensor> w_cuda;
  if (device.is_cuda()) {
    w_cuda.reserve(param_count);
    for (const auto& param : w) {
      auto cuda_tensor = torch::empty_like(param, torch::kCUDA);
      cudaMemcpyAsync(
        cuda_tensor.data_ptr<float>(),
        param.data_ptr<float>(),
        param.numel() * sizeof(float),
        cudaMemcpyHostToDevice
      );
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
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(learnRate).momentum(0.5));

  std::cout << "Training model for round " << round << "\n";

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, subset_size);
    //train(epoch, model, device, *train_loader_default, optimizer, train_dataset_size_default);
    if (worker_id == 0 && round % 10 == 0) {
      Logger::instance().log("Testing model after training round " + std::to_string(round) + "\n");
      test(model, device, *test_loader, test_dataset_size);
    }
  }
  
  std::vector<torch::Tensor> result;
  result.reserve(param_count);
  if (device.is_cuda()) {
    for (size_t i = 0; i < params.size(); ++i) {
      // Subtract on GPU
      auto update = params[i].clone().detach() - w_cuda[i];
      
      // Copy result to CPU
      auto cpu_update = torch::empty_like(update, torch::kCPU);
      cudaMemcpyAsync(
          cpu_update.data_ptr<float>(),
          update.data_ptr<float>(),
          update.numel() * sizeof(float),
          cudaMemcpyDeviceToHost
      );
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

  Logger::instance().log("Weight updates:\n");
  printTensorSlices(result, 0, 5);
  
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