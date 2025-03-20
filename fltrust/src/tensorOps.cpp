#include "../include/tensorOps.hpp"
#include "../include/logger.hpp"

#include <iostream>
#include <vector>
#include <sstream>

TensorOps::TensorOps(){}
TensorOps::~TensorOps(){}

// Function to flatten a vector of tensors into a single contiguous tensor
torch::Tensor TensorOps::flatten_tensor_vector(const std::vector<torch::Tensor>& tensor_vec) {
    std::vector<torch::Tensor> flattened_tensors;

    for (const auto& tensor : tensor_vec) {
        flattened_tensors.push_back(tensor.reshape(-1));
    }

    auto all_tensors = torch::cat(flattened_tensors).contiguous();
    
    return all_tensors;
}

// Function to reconstruct a vector of tensors from a flattened tensor
// Uses locally known reference tensor, so the client must do one dummy run of the model to get the reference W
std::vector<torch::Tensor> TensorOps::reconstruct_tensor_vector(
    const torch::Tensor& flat_tensor, 
    const std::vector<torch::Tensor>& reference_tensors) {
    
    std::vector<torch::Tensor> result;
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

void TensorOps::printTensorSlices(const std::vector<torch::Tensor>& model_weights, int start_idx, int end_idx) {
    std::ostringstream oss;
    oss << "Tensor slices:" << "\n";
    
    for (size_t i = 0; i < model_weights.size(); ++i) {
        const torch::Tensor& tensor = model_weights[i];
        
        // Handle the end_idx
        if (end_idx < 0 || end_idx > tensor.numel()) {
            end_idx = tensor.numel();
            oss << "Max length of this tensor is: " << end_idx << "\n";
        }
        
        // Make sure indices are valid
        int actual_start = std::min(start_idx, static_cast<int>(tensor.numel() - 1));
        int actual_end = std::min(end_idx, static_cast<int>(tensor.numel()));
        
        oss << "Tensor " << i << " (shape: " << tensor.sizes() << ") slice [" 
            << actual_start << ":" << actual_end << "]:";
                  
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