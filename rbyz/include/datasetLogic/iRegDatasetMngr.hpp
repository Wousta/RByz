#pragma once
#include "entities.hpp"
#include "structs.hpp"
#include <random>
#include <torch/torch.h>
#include <vector>

class IRegDatasetMngr {
public:
  const int worker_id;
  TrainInputParams t_params;
  const int n_clients;
  const int64_t kTrainBatchSize;
  const int64_t kNumberOfEpochs;
  const int64_t kTestBatchSize = 1000;
  const int64_t kLogInterval = 10;
  int64_t subset_size = 0;
  size_t test_dataset_size;
  double learning_rate;
  float loss;
  float test_loss;
  float error_rate;
  float train_accuracy = 0.0; // Initialize train accuracy
  float test_accuracy = 0.0;  // Initialize test accuracy
  RegTrainData data_info;
  ForwardPassData f_pass_data;

  IRegDatasetMngr(int worker_id, TrainInputParams &t_params)
      : worker_id(worker_id), t_params(t_params),
        n_clients(t_params.n_clients), kTrainBatchSize(t_params.batch_size),
        kNumberOfEpochs(t_params.epochs), learning_rate(t_params.local_learn_rate) {
          if (worker_id == 0) {
            subset_size = t_params.srvr_subset_size;
          } else {
            subset_size = t_params.clnt_subset_size;
          }
        }
  virtual ~IRegDatasetMngr() = default;

  virtual std::vector<torch::Tensor>
  runTraining(int round, const std::vector<torch::Tensor> &w) = 0;
  virtual void runTesting() = 0;
  virtual void runInference(const std::vector<torch::Tensor> &w) = 0;

  virtual std::vector<torch::Tensor> getInitialWeights() = 0;
  virtual std::vector<torch::Tensor>
  updateModelParameters(const std::vector<torch::Tensor> &w) = 0;
  virtual torch::Device getDevice() = 0;
  virtual std::vector<size_t> getClientsSamplesCount(uint32_t clnt_subset_size,
                                                     uint32_t srvr_subset_size,
                                                     uint32_t dataset_size) = 0;

  // Sample access methods
  virtual uint64_t getSampleOffset(size_t image_idx) = 0;
  virtual void *getSample(size_t image_idx) = 0;
  virtual uint32_t *getOriginalIndex(size_t image_idx) = 0;
  virtual int64_t *getLabel(size_t image_idx) = 0;
  // virtual uint8_t *getImage(size_t image_idx) = 0; // UINT8CHANGE
  virtual float *getImage(size_t image_idx) = 0;

  // Label flipping attacks
  virtual void flipLabelsRandom(float flip_ratio, std::mt19937 &rng) = 0;
  virtual void flipLabelsTargeted(int source_label, int target_label,
                                  float flip_ratio, std::mt19937 &rng) = 0;
  virtual void corruptImagesRandom(float flip_ratio, std::mt19937 &rng) = 0;
  virtual std::vector<size_t> findSamplesWithLabel(int label) = 0;
};