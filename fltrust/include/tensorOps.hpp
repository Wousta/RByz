#ifndef TENSOR_OPS_HPP
#define TENSOR_OPS_HPP


#include <vector>

class TensorOps {

  public:
  TensorOps();
  ~TensorOps();

  // Functions to execute Tensor operations
  torch::Tensor flatten_tensor_vector(const std::vector<torch::Tensor>& tensor_vec);
  std::vector<torch::Tensor> reconstruct_tensor_vector(
    const torch::Tensor& flat_tensor, 
    const std::vector<torch::Tensor>& reference_tensors);
  
  void printTensorSlices(const std::vector<torch::Tensor>& model_weights,  int start_idx = 0, int end_idx = -1);

};

#endif