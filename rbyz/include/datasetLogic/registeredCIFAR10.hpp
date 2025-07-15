// Code from:
// https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/popular/blitz/training_a_classifier/include/cifar10.h

// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include "datasetLogic/structs.hpp"
#include <cstddef>
#include <fstream>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

// CIFAR10 dataset
// based on:
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/mnist.h.
class RegCIFAR10 : public torch::data::datasets::Dataset<RegCIFAR10> {
public:
  // The mode in which the dataset is loaded
  enum Mode { kTrain, kTest, kBuild };

  // Loads the CIFAR10 dataset from the `root` path.
  //
  // The supplied `root` path should contain the *content* of the unzipped
  // CIFAR10 dataset (binary version), available from
  // http://www.cs.toronto.edu/~kriz/cifar.html.
  explicit RegCIFAR10(const std::string &root, Mode mode = Mode::kTrain);
  explicit RegCIFAR10(RegTrainData& data_info, std::unordered_map<size_t, size_t> index_map);

  // Returns the `Example` at the given `index`.
  torch::data::Example<> get(size_t index) override;

  // Returns the size of the dataset.
  torch::optional<size_t> size() const override;

  // Returns true if this is the training subset of CIFAR10.
  bool is_train() const noexcept;

  // Returns all images stacked into a single tensor.
  const torch::Tensor &images() const;

  // Returns all targets stacked into a single tensor.
  const torch::Tensor &targets() const;

private:
  torch::Tensor images_;
  torch::Tensor targets_;
  Mode mode_;

  RegTrainData *data_info;
  std::unordered_map<size_t, size_t> index_map;
  std::vector<int> shuffled_indices;
  torch::TensorOptions options;
  size_t num_samples = 0;
  size_t sample_size = 0;
  size_t index_size = 0;
  size_t label_size = 0;
  size_t image_size = 0;
};
