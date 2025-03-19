#include "../include/tensorOps.hpp"
#include "../include/logger.hpp"

#include <iostream>
#include <vector>
#include <sstream>

TensorOps::TensorOps(){}
TensorOps::~TensorOps(){}

void TensorOps::printTensorSlices(const std::vector<torch::Tensor>& model_weights, int start_idx, int end_idx) {
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
    
    // Log the entire output using your logger
    Logger::instance().log(oss.str());
}