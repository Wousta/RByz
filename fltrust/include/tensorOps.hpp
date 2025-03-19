#ifndef TENSOR_OPS_HPP
#define TENSOR_OPS_HPP


#include <vector>

class TensorOps {

  public:
  TensorOps();
  ~TensorOps();

  // Functions to execute Tensor operations
  void printTensorSlices(const std::vector<torch::Tensor>& model_weights,  int start_idx = 0, int end_idx = -1);

};

#endif