#pragma once

#include "datasetLogic/mnistTrain.hpp"
#include <vector>

std::vector<torch::Tensor> no_byz(
  const std::vector<torch::Tensor>& v,
  Net net,
  int lr,
  int f,
  torch::Device device
);

std::vector<torch::Tensor> trim_attack(
  const std::vector<torch::Tensor>& v,
  Net net,
  int lr,
  int f,
  torch::Device device
);

std::vector<torch::Tensor> krum_attack(
  const std::vector<torch::Tensor>& v,
  Net net,
  int lr,
  int f,
  torch::Device device
);