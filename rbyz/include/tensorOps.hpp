#pragma once

#include <vector>
#include <torch/torch.h>

namespace tops {

inline torch::Tensor flatten_tensor_vector(const std::vector<torch::Tensor>& tensor_vec) {
    std::vector<torch::Tensor> flattened_tensors;
    flattened_tensors.reserve(tensor_vec.size());
    
    for (const auto& tensor : tensor_vec) {
        flattened_tensors.push_back(tensor.reshape(-1));
    }
    
    return torch::cat(flattened_tensors).contiguous();
}

std::vector<torch::Tensor> reconstruct_tensor_vector(
    const torch::Tensor& flat_tensor, 
    const std::vector<torch::Tensor>& reference_tensors);

void printTensorSlices(const std::vector<torch::Tensor>& model_weights, 
                      int start_idx, int end_idx);

void memcpyTensorVec(float *dest, std::vector<torch::Tensor> &src, size_t size);
void memcpyTensor(float *dest, torch::Tensor &src, size_t size);
void writeToTensorVec(std::vector<torch::Tensor> &dest, float *src, size_t size);

} // namespace tops
