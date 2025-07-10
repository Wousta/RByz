#pragma once

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "entities.hpp"
#include <vector>

std::vector<torch::Tensor> no_byz(
  const std::vector<torch::Tensor>& v,
  int lr,
  int f,
  torch::Device device
);

std::vector<torch::Tensor> trim_attack(
  const std::vector<torch::Tensor>& v,
  int lr,
  int f,
  torch::Device device
);

std::vector<torch::Tensor> krum_attack(
  const std::vector<torch::Tensor>& v,
  int lr,
  int f,
  torch::Device device
);

void data_poison_attack(bool use_mnist, TrainInputParams &t_params, IRegDatasetMngr &mngr);