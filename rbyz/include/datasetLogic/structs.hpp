#pragma once

#include <cuda_runtime.h>

/**
 * Memory layout per sample:
 * [1 uint32_t (original index)][1 int64_t (label)][784 floats (pixels)]
 */
struct RegTrainData {
  void *reg_data = nullptr;
  size_t reg_data_size = 0;
  size_t num_samples = 0;
  size_t image_size = 0;      
  const size_t index_size = sizeof(uint32_t);
  const size_t label_size = sizeof(int64_t);
  
  size_t get_sample_size() const {
    if (image_size == 0) {
      throw std::runtime_error("Image size must be initialized in RegTrainData");
    }
    return index_size + label_size + image_size;
  }
};

struct ForwardPassData {
  int num_batches = 0;
  float *forward_pass = nullptr;
  uint32_t *forward_pass_indices = nullptr;
  size_t forward_pass_mem_size = 0;
  size_t forward_pass_indices_mem_size = 0;
  const size_t values_per_sample = 2;           // 2 values (error and loss) in the forward pass
  const size_t bytes_per_value = sizeof(float); 
};

struct ForwardPassBuffer {
  std::vector<torch::Tensor> outputs;
  std::vector<torch::Tensor> targets;
  std::vector<torch::Tensor> losses;
};