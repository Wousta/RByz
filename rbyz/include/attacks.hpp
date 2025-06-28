#pragma once

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "entities.hpp"
#include <vector>

#define NO_ATTACK 0
#define RANDOM_FLIP 1
#define TARGETED_FLIP_1 2
#define TARGETED_FLIP_2 3
#define TARGETED_FLIP_3 4

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

void label_flip_attack(bool use_mnist, TrainInputParams &t_params, IRegDatasetMngr &mngr);