#pragma once
#include "entities.hpp"
#include "structs.hpp"
#include <random>
#include <torch/torch.h>
#include <vector>

#define INACTIVE -1

class IRegDatasetMngr {
public:
  const int worker_id;
  TrainInputParams t_params;
  const int n_clients;
  const int64_t kTrainBatchSize;
  const int64_t kTestBatchSize = 1000;
  const int64_t kLogInterval = 10;
  const int overwrite_poisoned;
  int64_t kNumberOfEpochs;
  int64_t subset_size = 0;
  size_t test_dataset_size;
  size_t train_dataset_size;
  double learn_rate;
  float loss;
  float test_loss;
  float error_rate;
  float train_accuracy = 0.0; // Initialize train accuracy
  float test_accuracy = 0.0;  // Initialize test accuracy
  float src_class_recall = 0.0;
  int src_class = INACTIVE;  // Source class for targeted attacks, INACTIVE if not applicable
  int target_class = INACTIVE;
  int missclassed_samples = 0;
  bool attack_is_targeted_flip = false;
  RegTrainData data_info;
  ForwardPassData f_pass_data;

  IRegDatasetMngr(int worker_id, TrainInputParams &t_params)
      : worker_id(worker_id), t_params(t_params),
        n_clients(t_params.n_clients), kTrainBatchSize(t_params.batch_size), overwrite_poisoned(t_params.overwrite_poisoned),
        kNumberOfEpochs(t_params.epochs), learn_rate(t_params.local_learn_rate) {
          if (worker_id == 0) {
            subset_size = t_params.srvr_subset_size;
          } else {
            subset_size = t_params.clnt_subset_size;
          }
        }
  virtual ~IRegDatasetMngr() = default;

  virtual void runTraining() = 0;
  virtual void runTesting() = 0;
  virtual void runInference(const std::vector<torch::Tensor> &w) = 0;
  virtual void renewDataset(float proportion = 1.0, std::optional<int> seed = std::nullopt) = 0;

  virtual std::vector<torch::Tensor>
  calculateUpdate(const std::vector<torch::Tensor> &w) = 0;
  virtual std::vector<torch::Tensor>
  updateModelParameters(const std::vector<torch::Tensor> &w) = 0;
  virtual std::vector<torch::Tensor> getInitialWeights() = 0;
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