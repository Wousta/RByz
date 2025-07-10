#include "tensorOps.hpp"
#include "logger.hpp"

#include <iostream>
#include <vector>
#include <sstream>

namespace tops {

/**
 * Reconstruct a vector of tensors from a flattened tensor.
 * @param flat_tensor: The flattened tensor.
 * @param reference_tensors: The reference tensors to use for reshaping.
 * @return The reconstructed vector of tensors.
 */
std::vector<torch::Tensor> reconstruct_tensor_vector(
    const torch::Tensor& flat_tensor, 
    const std::vector<torch::Tensor>& reference_tensors) {
    
    std::vector<torch::Tensor> result;
    result.reserve(reference_tensors.size());
    int64_t offset = 0;
    
    // Reconstruct each tensor using the shapes from reference_tensors
    for (const auto& ref_tensor : reference_tensors) {
        auto shape = ref_tensor.sizes().vec();
        int64_t num_elements = ref_tensor.numel();
        
        // Extract and reshape the tensor
        torch::Tensor tensor_portion = flat_tensor.slice(0, offset, offset + num_elements);
        result.push_back(tensor_portion.reshape(shape));
        
        offset += num_elements;
    }
    
    return result;
}

/**
 * Print slices of tensors in a vector.
 * @param model_weights: The vector of tensors to print.
 * @param start_idx: The starting index of the slice.
 * @param end_idx: The ending index of the slice.
 */
void printTensorSlices(const std::vector<torch::Tensor>& model_weights, int start_idx, int end_idx) {
    std::ostringstream oss;
    oss << "Tensor slices:" << "\n";
    
    for (size_t i = 0; i < model_weights.size(); ++i) {
        const torch::Tensor& tensor = model_weights[i];
        
        // Handle the end_idx
        if (end_idx < 0 || end_idx > tensor.numel()) {
            end_idx = tensor.numel();
        }
        
        // Make sure indices are valid
        int actual_start = std::min(start_idx, static_cast<int>(tensor.numel() - 1));
        int actual_end = std::min(end_idx, static_cast<int>(tensor.numel()));
        
        oss << "Tensor " << i << " (shape: " << tensor.sizes() << ") slice [" 
            << actual_start << ":" << actual_end << "]:\n";
                  
        // For 1D tensors, we can directly slice
        if (tensor.dim() == 1) {
            auto slice = tensor.slice(0, actual_start, actual_end);
            oss << slice << " ";
        } 
        // For multidimensional tensors, we need to flatten first to get continuous elements
        else {
            auto flattened = tensor.flatten();
            auto slice = flattened.slice(0, actual_start, actual_end);
            oss << slice << " ";
            
            // Optionally show the original shape
            oss << "(Original tensor is " << tensor.dim() << "D)";
        }
        
        oss << "...\n";
    }
    
    Logger::instance().log(oss.str());
}

void memcpyTensorVec(float *dest, std::vector<torch::Tensor> &src, size_t size) {
  if (src.empty()) {
    Logger::instance().log("No tensors to copy, returning.\n");
    return;
  }

  auto all_tensors = flatten_tensor_vector(src);
  size_t total_bytes = all_tensors.numel() * sizeof(float);

  if (total_bytes != size) {
    throw std::runtime_error("[memcpyTensorVec] Size mismatch: expected " + std::to_string(size) +
                             ", but got " + std::to_string(total_bytes));
  }

  float *src_float = all_tensors.data_ptr<float>();
  std::memcpy(dest, src_float, total_bytes);
}

void memcpyTensor(float *dest, torch::Tensor &src, size_t size) {
  if (src.numel() == 0) {
    Logger::instance().log("Source tensor is empty, returning.\n");
    return;
  }

  size_t total_bytes = src.numel() * sizeof(float);
  if (total_bytes != size) {
    throw std::runtime_error("[memcpyTensor] Size mismatch: expected " + std::to_string(size) +
                             ", but got " + std::to_string(total_bytes));
  }

  float *src_float = src.data_ptr<float>();
  std::memcpy(dest, src_float, total_bytes);
}

void writeToTensorVec(std::vector<torch::Tensor> &dest, float *src, size_t size) {
  if (dest.empty()) {
    Logger::instance().log("No tensors to read, returning.\n");
    return;
  }

  size_t num_elements = size / sizeof(float);
  torch::Tensor flat_tensor = torch::from_blob(
      src, {static_cast<long>(num_elements)}, torch::kFloat32
  ).clone();
  dest = reconstruct_tensor_vector(flat_tensor, dest);
}

} // namespace tops