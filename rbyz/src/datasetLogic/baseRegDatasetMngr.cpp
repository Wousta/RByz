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