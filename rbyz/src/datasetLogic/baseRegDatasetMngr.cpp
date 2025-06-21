#include "datasetLogic/baseRegDatasetMngr.hpp"
#include "logger.hpp"
#include "tensorOps.hpp"

BaseRegDatasetMngr::BaseRegDatasetMngr(int worker_id, int num_workers,
                                       int64_t subset_size,
                                       std::unique_ptr<NNet> net)
    : worker_id(worker_id), num_workers(num_workers), subset_size(subset_size),
      model(std::move(net)), device(init_device()) {
  torch::manual_seed(1);
  model->to(device);

  if (device.is_cuda()) {
    cudaStreamCreate(&memcpy_stream_A);
    cudaStreamCreate(&memcpy_stream_B);
  }
}

BaseRegDatasetMngr::~BaseRegDatasetMngr() {
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
  }
}

torch::Device BaseRegDatasetMngr::init_device() {
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

std::vector<torch::Tensor> BaseRegDatasetMngr::getInitialWeights() {
  std::vector<torch::Tensor> initialWeights;
  for (const auto &param : model->parameters()) {
    // Must be at CPU at the beginning
    initialWeights.push_back(param.clone().detach().to(torch::kCPU));
  }

  Logger::instance().log("Obtained initial random weights from model\n");
  printTensorSlices(initialWeights, 0, 5);

  return initialWeights;
}

template <typename DataLoader>
void BaseRegDatasetMngr::test(DataLoader& data_loader, size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model->eval();
  double test_loss = 0;
  int32_t correct = 0;
  int32_t total_samples = 0;  // Track actual samples processed
  
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model->forward(data);
    test_loss += torch::nll_loss(output,
                                 targets,
                                 /*weight=*/{},
                                 torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
    total_samples += targets.size(0);  // Add actual batch size
  }

  test_loss /= total_samples;  // Use actual samples processed
  setTestLoss(test_loss);  // Set the test loss
  setTestAccuracy(static_cast<float>(correct) / total_samples);  // Use actual samples
  
  std::ostringstream oss;
  oss << "\n  Test set: Average loss: " << std::fixed << std::setprecision(4) << test_loss
      << " | Accuracy: " << std::fixed << std::setprecision(3)
      << getTestAccuracy() 
      << " (" << correct << "/" << total_samples << ")";  // Show actual counts for verification
  Logger::instance().log(oss.str() + "\n");
  Logger::instance().log("  Testing done\n");
}

std::vector<torch::Tensor> BaseRegDatasetMngr::updateModelParameters(const std::vector<torch::Tensor>& w) {
  std::vector<torch::Tensor> params = model->parameters();
  size_t param_count = params.size();
  std::vector<torch::Tensor> w_cuda;
  
  if (device.is_cuda()) {
    w_cuda.reserve(param_count);
    for (const auto& param : w) {
      auto cuda_tensor = torch::empty_like(param, torch::kCUDA);
      cudaMemcpyAsync(cuda_tensor.data_ptr<float>(),
                      param.data_ptr<float>(),
                      param.numel() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      memcpy_stream_A);
      w_cuda.push_back(cuda_tensor);
    }

    cudaDeviceSynchronize();
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w_cuda[i]);
    }
    return w_cuda;
  } else {
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].data().copy_(w[i]);
    }
    return w;
  }
}