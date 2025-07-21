#pragma once

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "entities.hpp"
#include <vector>

#define NUM_CLASSES 10 // Number of classes of CIFAR-10 and MNIST

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