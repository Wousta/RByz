#include "../include/mnistTrain.hpp"
#include "../include/logger.hpp"
#include "../include/tensorOps.hpp"


#include <cstddef>
#include <cstdio>
#include <iostream> 
#include <string>
#include <vector>

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
  if(worker_id == 0) {
    torch::manual_seed(1);
  }
  else {
    torch::manual_seed(2);
  }

  if (torch::cuda::is_available()) {
    Logger::instance().log("CUDA available! Training on GPU.\n");
    device_type = torch::kCUDA;
  } else {
    Logger::instance().log("Training on CPU.\n");
    device_type = torch::kCPU;
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
  
  // Extract model weights after training
  std::vector<torch::Tensor> model_weights;
  for (const auto& param : model.parameters()) {
    // Clone the parameter to ensure we're not just storing references
    model_weights.push_back(param.clone().detach());
  }

  return model_weights;
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

  // Extract model weights after training
  std::vector<torch::Tensor> model_weights;
  for (const auto& param : model.parameters()) {
    // Clone the parameter to ensure we're not just storing references
    //model_weights.push_back(param.clone().detach());
    auto shape = param.sizes();
    auto random_tensor = torch::rand(shape, param.options());
    random_tensor = random_tensor * 0.1;
    model_weights.push_back(random_tensor);
  }

  return model_weights;
}

void MnistTrain::testModel() {
  test(model, device, *test_loader, test_dataset_size);
}

void MnistTrain::saveModelState(const std::vector<torch::Tensor>& w, const std::string& filename) {
  try {
    Logger::instance().log("Saving model state to " + filename + "...\n");
    torch::save(w, filename);
    Logger::instance().log("Model state saved successfully.\n");
  } catch (const std::exception& e) {
    Logger::instance().log("Error saving model state: " + std::string(e.what()) + "\n");
  }
}

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