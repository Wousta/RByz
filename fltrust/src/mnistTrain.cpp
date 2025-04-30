#include "../include/mnistTrain.hpp"
#include "../include/logger.hpp"
#include "../include/tensorOps.hpp"


#include <cstddef>
#include <cstdio>
#include <iostream> 
#include <string>
#include <vector>
#include <cuda_runtime.h>

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
  auto indices_tensor = torch::randperm(dataset_size);

  int64_t start = (worker_id - 1) * subset_size;
  if(worker_id == 0) {
    start = 0;
  }
  
  int64_t end = start + subset_size;
  if (end > dataset_size) {
    end = dataset_size;
  }

  auto subset_tensor = indices_tensor.slice(0, start, end);

  std::vector<size_t> subset_indices(subset_size);
  for (int64_t i = 0; i < subset_size; ++i) {
    subset_indices[i] = static_cast<size_t>(subset_tensor[i].item<int64_t>());
  }

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
  Logger::instance().log("Training in device type: " + std::string(device.is_cuda() ? "GPU" : "CPU") + "\n");
  model.train();
  size_t batch_idx = 0;
  int32_t correct = 0;
  size_t total = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    output = model.forward(data);
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

std::vector<torch::Tensor> MnistTrain::testOG() {
  int local_subset_size = 2;
  SubsetSampler train_sampler_local = get_subset_sampler(worker_id, train_dataset_size, local_subset_size);

  auto train_loader_local = torch::data::make_data_loader(
    train_dataset,
    train_sampler_local,
    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));

  torch::optim::SGD optimizer(
    model.parameters(), torch::optim::SGDOptions(learnRate).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader_local, optimizer, local_subset_size);
  }

  size_t param_count = model.parameters().size();
  std::vector<torch::Tensor> cpu_tensors;
  cpu_tensors.reserve(param_count);
  std::vector<at::Tensor> cuda_tensors;
  cuda_tensors.reserve(param_count);
  if (device.is_cuda()) {
    for (const auto& param : model.parameters()) {
      auto shape = param.sizes();
      auto random_tensor = torch::rand(shape, 
                                      torch::TensorOptions()
                                      .device(device)
                                      .dtype(param.dtype()));
      random_tensor = random_tensor * 0.1;
      
      // Push tensor to vector to keep it in scope while it is asynchronously copied
      cuda_tensors.push_back(random_tensor);
      auto cpu_tensor = torch::empty_like(random_tensor, torch::kCPU);
      cudaMemcpyAsync(
        cpu_tensor.data_ptr<float>(),
        random_tensor.data_ptr<float>(),
        random_tensor.numel() * sizeof(float),
        cudaMemcpyDeviceToHost
      );
      cpu_tensors.push_back(cpu_tensor);
    }

    // Synchronize to ensure all async copies are done
    cudaDeviceSynchronize();
  } else {
    for (const auto& param : model.parameters()) {
      auto shape = param.sizes();
      auto random_tensor = torch::rand(shape, 
                                     torch::TensorOptions()
                                     .device(device)
                                     .dtype(param.dtype()));
      random_tensor = random_tensor * 0.1;
      cpu_tensors.push_back(random_tensor);
    }
  }

  return cpu_tensors;
}

void MnistTrain::testModel() {
  test(model, device, *test_loader, test_dataset_size);
}