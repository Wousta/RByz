#include "datasetLogic/baseMnistTrain.hpp"
#include "global/logger.hpp"
#include "tensorOps.hpp"

#include <random>

BaseMnistTrain::BaseMnistTrain(int worker_id, int num_workers, int64_t subset_size)
    : worker_id(worker_id),
      num_workers(num_workers),
      subset_size(subset_size),
      device(init_device()),
      test_dataset(
          torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
              .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
              .map(torch::data::transforms::Stack<>())) {
  torch::manual_seed(1);
  model.to(device);

  test_dataset_size = test_dataset.size().value();
  auto test_loader_temp = torch::data::make_data_loader(
      test_dataset, torch::data::DataLoaderOptions().batch_size(kTestBatchSize));
  test_loader = std::move(test_loader_temp);

  buildLabelToIndicesMap();

  if (device.is_cuda()) {
    cudaStreamCreate(&memcpy_stream_A);
    cudaStreamCreate(&memcpy_stream_B);
  }
}

BaseMnistTrain::~BaseMnistTrain() {
  if (device.is_cuda()) {
    cudaError_t err_A, err_B;
    err_A = cudaStreamDestroy(memcpy_stream_A);
    err_B = cudaStreamDestroy(memcpy_stream_B);
    if (err_A != cudaSuccess) {
      std::cerr << "Error destroying stream A: " << cudaGetErrorString(err_A) << std::endl;
    }
    if (err_B != cudaSuccess) {
      std::cerr << "Error destroying stream B: " << cudaGetErrorString(err_B) << std::endl;
    }
  }
}

torch::Device BaseMnistTrain::init_device() {
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

std::vector<torch::Tensor> BaseMnistTrain::getInitialWeights() {
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

SubsetSampler BaseMnistTrain::get_subset_sampler(int worker_id_arg,
                                                size_t dataset_size_arg,
                                                int64_t subset_size_arg) {

  // Allocate indices to workers
  float srvr_proportion = static_cast<float>(SRVR_SUBSET_SIZE) / dataset_size_arg;
  float clnt_proportion = (1 - srvr_proportion) / (num_workers - 1);

  std::vector<size_t> worker_indices;
  for (const auto& [label, indices] : label_to_indices) {
    size_t total_samples = indices.size();
    size_t samples_srvr = static_cast<size_t>(std::ceil(total_samples * srvr_proportion));
    size_t samples_clnt = static_cast<size_t>(std::floor(total_samples * clnt_proportion));

    size_t start;
    size_t end;
    if (worker_id_arg == 0) {
      start = 0;  // Server gets the first portion
      end = samples_srvr;
    } else {
      start = static_cast<size_t>(std::ceil(samples_srvr + (worker_id_arg - 1) * samples_clnt));
      if (worker_id_arg == num_workers - 1) {
        end = total_samples;  // Last worker gets the remaining samples
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
  if (worker_indices.size() > subset_size_arg) {
    worker_indices.resize(subset_size_arg);
  }

  Logger::instance().log(
      "Worker " + std::to_string(worker_id) +
      " using stratified indices of size: " + std::to_string(worker_indices.size()) + "\n");

  return SubsetSampler(worker_indices);
}

template <typename DataLoader>
void BaseMnistTrain::test(Net& model,
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
    test_loss += torch::nll_loss(output,
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
      << static_cast<double>(correct) / dataset_size;
  Logger::instance().log(oss.str() + "\n");
  Logger::instance().log("Testing done\n");
}

void BaseMnistTrain::saveModelState(const std::vector<torch::Tensor>& w, const std::string& filename) {
  try {
    Logger::instance().log("Saving model state to " + filename + "...\n");
    torch::save(w, filename);
    Logger::instance().log("Model state saved successfully.\n");
  } catch (const std::exception& e) {
    Logger::instance().log("Error saving model state: " + std::string(e.what()) + "\n");
  }
}

std::vector<torch::Tensor> BaseMnistTrain::loadModelState(const std::string& filename) {
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

void BaseMnistTrain::copyModelParameters(const Net& source_model) {
    auto source_params = source_model.parameters();
    auto dest_params = model.parameters();
    
    if (source_params.size() == dest_params.size()) {
        for (size_t i = 0; i < source_params.size(); ++i) {
            dest_params[i].data().copy_(source_params[i].data().to(device));
        }
    } else {
        throw std::runtime_error("Model parameter count mismatch during parameter copy");
    }
}

std::vector<size_t> BaseMnistTrain::getClientsSamplesCount() {
  Logger::instance().log("Getting samples count for each client\n");
  std::vector<size_t> samples_count(num_workers, 0);

  // Allocate indices to workers
  float srvr_proportion = static_cast<float>(SRVR_SUBSET_SIZE) / DATASET_SIZE;
  float clnt_proportion = (1 - srvr_proportion) / (num_workers - 1);
  for (size_t i = 1; i < num_workers; i++) {
    std::vector<size_t> worker_indices;
    for (const auto& [label, indices] : label_to_indices) {
      size_t total_samples = indices.size();
      size_t samples_clnt = static_cast<size_t>(std::floor(total_samples * clnt_proportion));
      size_t samples_srvr = static_cast<size_t>(std::ceil(total_samples * srvr_proportion));


      size_t start = static_cast<size_t>(std::ceil(samples_srvr + (i - 1) * samples_clnt));
      size_t end;
      if (i == num_workers - 1) {
        end = total_samples;  // Last worker gets the remaining samples
      } else {
        end = start + samples_clnt;
      }

      // Add the worker's portion of indices for this label
      worker_indices.insert(worker_indices.end(), indices.begin() + start, indices.begin() + end);
    }

    if (worker_indices.size() > CLNT_SUBSET_SIZE) {
      worker_indices.resize(CLNT_SUBSET_SIZE);
    }

    size_t worker_samples_count = worker_indices.size();
    samples_count[i] = worker_samples_count;
    Logger::instance().log("Worker " + std::to_string(i) +
                            " using stratified indices of size: " +
                            std::to_string(worker_samples_count) + "\n");
  }

  return samples_count;
}

void BaseMnistTrain::buildLabelToIndicesMap() {
  auto dataset = torch::data::datasets::MNIST(kDataRoot).
    map(torch::data::transforms::Normalize<>(0.1307, 0.3081)).
    map(torch::data::transforms::Stack<>());

  size_t index = 0;
  // Create a DataLoader to iterate over the dataset
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      dataset, /*batch size*/ 1);

  for (const auto& example : *data_loader) {
    int64_t label = example.target.item<int64_t>();
    label_to_indices[label].push_back(index);
    ++index;
  }
}

// Explicit template instantiation
template void BaseMnistTrain::test<torch::data::StatelessDataLoader<BaseMnistTrain::DatasetType, torch::data::samplers::RandomSampler>>(
    Net&, 
    torch::Device,
    torch::data::StatelessDataLoader<BaseMnistTrain::DatasetType, torch::data::samplers::RandomSampler>&,
    size_t);