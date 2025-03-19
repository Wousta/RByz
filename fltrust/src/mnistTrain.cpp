#include "../include/mnistTrain.hpp"
#include "../include/logger.hpp"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int counter = 1;

// The model instance
Net model;

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 1;

// The batch size for testing.
const int64_t kTestBatchSize = 1;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1;

const int64_t learnRate = 1;

std::vector<torch::Tensor> runMNISTTrainDummy(std::vector<torch::Tensor>& w) {
  std::cout << "Running dummy MNIST training\n";
  
  for(size_t i = 0; i < w.size(); i++) {
    w[i] = w[i] + 1;  // element-wise addition of 1
  }
  
  return w;
}

template <typename DataLoader>
void train(
    size_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
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
  // Logger::instance().log(
  //     "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
  //     test_loss,
  //     static_cast<double>(correct) / dataset_size);
  Logger::instance().log("Testing donete\n");
}

std::vector<torch::Tensor> runMnistTrain(const std::vector<torch::Tensor>& w) {
  torch::manual_seed(1);
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);
  Net model;
  model.to(device);
  
  // Update model parameters with input weights if sizes match
  auto params = model.parameters();
  if (w.size() == params.size()) {
    std::cout << "Updating model parameters with input weights." << std::endl;
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
  
  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();

  // Create a subset random sampler with these indices
  auto sampler = torch::data::samplers::RandomSampler(train_dataset_size);

  // Create the data loader with the sampler
  auto train_loader = torch::data::make_data_loader(
      train_dataset,
      sampler,
      torch::data::DataLoaderOptions().batch_size(1).workers(2)
  );

  // auto train_loader =
  //     torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
  //         std::move(train_dataset), kTrainBatchSize);
  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(learnRate).momentum(0.5));
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
  
  // Extract model weights after training
  std::vector<torch::Tensor> model_weights;
  for (const auto& param : model.parameters()) {
    // Clone the parameter to ensure we're not just storing references
    model_weights.push_back(param.clone().detach());
  }
  
  return model_weights;
}


std::vector<torch::Tensor> testOG() {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

  // Extract model weights after training
  std::vector<torch::Tensor> model_weights;
  for (const auto& param : model.parameters()) {
    // Clone the parameter to ensure we're not just storing references
    model_weights.push_back(param.clone().detach());
  }
  
  return model_weights;
}

std::vector<torch::Tensor> aggregateResults(
  const std::vector<torch::Tensor>& server_update,
  std::vector<float*>& client_weights,
  const std::vector<int>& polled_clients) {
  
    Logger::instance().log("Starting FLTrust aggregation...\n");
    
    // Get shapes from the model parameters
    std::vector<std::vector<int64_t>> param_shapes;
    std::vector<int64_t> param_sizes;
    
    for (const auto& param : model.parameters()) {
        param_shapes.push_back(param.sizes().vec());
        param_sizes.push_back(param.numel());
    }
    
    // Reshape the client data into tensors with proper shapes
    std::vector<std::vector<torch::Tensor>> client_updates(client_weights.size());
    
    // Process client updates
    for (int client_idx : polled_clients) {
        std::vector<torch::Tensor> client_tensors;
        size_t offset = 0;
        
        for (size_t i = 0; i < param_shapes.size(); i++) {
            auto tensor = torch::from_blob(
                client_weights[client_idx] + offset,
                {param_sizes[i]},
                torch::kFloat32
            ).clone();
            
            client_tensors.push_back(tensor.reshape(param_shapes[i]));
            offset += param_sizes[i];
        }
        
        client_updates[client_idx] = client_tensors;
    }
    
    // Initialize aggregated update with server tensor structure
    std::vector<torch::Tensor> aggregated_update = server_update;
    for (auto& tensor : aggregated_update) {
        tensor.zero_();
    }
    
    // Flatten the server update
    // Create a vector of tensors for cat
    std::vector<torch::Tensor> server_tensors_vec;
    for (const auto& tensor : server_update) {
        server_tensors_vec.push_back(tensor.flatten());
    }
    torch::Tensor flat_server_update = torch::cat(server_tensors_vec, 0).view({1, -1});
    float server_norm = torch::norm(flat_server_update).item<float>();
    
    // Calculate trust scores
    std::vector<float> trust_scores;
    std::vector<int> processed_clients;
    
    Logger::instance().log("Calculating trust scores for clients...\n");
    
    for (int client_idx : polled_clients) {
        // Flatten client update
        std::vector<torch::Tensor> client_tensors_vec;
        for (const auto& tensor : client_updates[client_idx]) {
            client_tensors_vec.push_back(tensor.flatten());
        }
        auto flat_client_update = torch::cat(client_tensors_vec, 0).view({1, -1});
        float client_norm = torch::norm(flat_client_update).item<float>();
        
        // Calculate cosine similarity
        float cos_sim = 0.0f;
        if (client_norm > 0 && server_norm > 0) {
            cos_sim = torch::cosine_similarity(flat_server_update, flat_client_update, 1).item<float>();
        }
        
        // ReLU function to clip negative similarities
        float trust = std::max(0.0f, cos_sim);
        trust_scores.push_back(trust);
        processed_clients.push_back(client_idx);
        
        Logger::instance().log("Client " + std::to_string(client_idx) + " trust score: " + std::to_string(trust) + "\n");
    }
    
    // Normalize trust scores
    float sum_trust = 0.0f;
    for (float trust : trust_scores) {
        sum_trust += trust;
    }
    
    std::vector<float> normalized_trust;
    if (sum_trust > 0) {
        for (float trust : trust_scores) {
            normalized_trust.push_back(trust / sum_trust);
        }
    } else {
        // Equal weights if all similarities are zero
        float equal_weight = 1.0f / processed_clients.size();
        normalized_trust = std::vector<float>(processed_clients.size(), equal_weight);
        Logger::instance().log("Warning: All trust scores are zero. Using equal weights.\n");
    }
    
    // Apply weighted aggregation
    for (size_t i = 0; i < processed_clients.size(); i++) {
        int client_idx = processed_clients[i];
        float weight = normalized_trust[i];
        
        Logger::instance().log("Applying weight " + std::to_string(weight) + " to client " + std::to_string(client_idx) + "\n");
        
        for (size_t j = 0; j < aggregated_update.size(); j++) {
            aggregated_update[j].add_(client_updates[client_idx][j] * weight);
        }
    }
    
    Logger::instance().log("FLTrust aggregation complete.\n");
    
    return aggregated_update;
}