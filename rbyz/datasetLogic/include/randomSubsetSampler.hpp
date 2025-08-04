#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <random>
#include <vector>

class RandomSubsetSampler : public torch::data::samplers::Sampler<std::vector<size_t>> {
 private:
  std::vector<size_t> indices_;
  std::vector<size_t> shuffled_indices_;
  size_t current_;
  std::mt19937 generator_;
  mutable std::vector<size_t> last_batch_indices_;

  void shuffle() {
    shuffled_indices_ = indices_;
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), generator_);
  }

 public:
  // Type alias required by the Sampler interface.
  using BatchRequestType = std::vector<size_t>;

  explicit RandomSubsetSampler(std::vector<size_t> indices,
                               std::optional<uint64_t> seed = std::nullopt)
      : indices_(std::move(indices)), current_(0) {
    if (seed.has_value()) {
      generator_.seed(seed.value());
    } else {
      generator_.seed(std::random_device{}());
    }
    shuffle();
  }

  // Reset the sampler with an optional new size.
  void reset(std::optional<size_t> new_size = std::nullopt) override {
    if (new_size.has_value()) {
      if (new_size.value() < indices_.size()) {
        indices_.resize(new_size.value());
      }
    }
    current_ = 0;
    shuffle();
  }

  // Returns the next batch.
  std::optional<BatchRequestType> next(size_t batch_size) override {
    BatchRequestType batch;
    last_batch_indices_.clear();

    while (batch.size() < batch_size && current_ < shuffled_indices_.size()) {
      size_t idx = shuffled_indices_[current_++];
      batch.push_back(idx);
      last_batch_indices_.push_back(idx);
    }

    if (batch.empty()) {
      return std::nullopt;
    }
    return batch;
  }

  const std::vector<size_t>& get_last_batch_indices() const { return last_batch_indices_; }

  // Serialize the sampler state.
  void save(torch::serialize::OutputArchive& archive) const override {
    // Convert indices_ to a tensor for serialization.
    torch::Tensor indices_tensor =
        torch::tensor(std::vector<int64_t>(indices_.begin(), indices_.end()), torch::kInt64);
    torch::Tensor shuffled_indices_tensor = torch::tensor(
        std::vector<int64_t>(shuffled_indices_.begin(), shuffled_indices_.end()), torch::kInt64);
    torch::Tensor current_tensor = torch::tensor(static_cast<int64_t>(current_), torch::kInt64);

    archive.write("indices", indices_tensor);
    archive.write("shuffled_indices", shuffled_indices_tensor);
    archive.write("current", current_tensor);

    // Save generator state (simplified approach)
    torch::Tensor seed_tensor =
        torch::tensor(static_cast<int64_t>(generator_.default_seed), torch::kInt64);
    archive.write("seed", seed_tensor);
  }

  // Deserialize the sampler state.
  void load(torch::serialize::InputArchive& archive) override {
    torch::Tensor indices_tensor, shuffled_indices_tensor, current_tensor, seed_tensor;

    archive.read("indices", indices_tensor);
    archive.read("shuffled_indices", shuffled_indices_tensor);
    archive.read("current", current_tensor);
    archive.read("seed", seed_tensor);

    // Restore indices_
    auto indices_numel = indices_tensor.numel();
    std::vector<int64_t> temp_indices(indices_numel);
    std::memcpy(temp_indices.data(), indices_tensor.data_ptr<int64_t>(),
                indices_numel * sizeof(int64_t));
    indices_.resize(indices_numel);
    for (size_t i = 0; i < indices_numel; ++i) {
      indices_[i] = static_cast<size_t>(temp_indices[i]);
    }

    // Restore shuffled_indices_
    auto shuffled_numel = shuffled_indices_tensor.numel();
    std::vector<int64_t> temp_shuffled(shuffled_numel);
    std::memcpy(temp_shuffled.data(), shuffled_indices_tensor.data_ptr<int64_t>(),
                shuffled_numel * sizeof(int64_t));
    shuffled_indices_.resize(shuffled_numel);
    for (size_t i = 0; i < shuffled_numel; ++i) {
      shuffled_indices_[i] = static_cast<size_t>(temp_shuffled[i]);
    }

    current_ = static_cast<size_t>(current_tensor.item<int64_t>());

    // Restore generator (simplified approach)
    uint64_t seed = static_cast<uint64_t>(seed_tensor.item<int64_t>());
    generator_.seed(seed);
  }

  const std::vector<size_t>& indices() const { return indices_; }
  const std::vector<size_t>& shuffled_indices() const { return shuffled_indices_; }

  // Set a new seed for the random number generator
  void set_seed(uint64_t seed) {
    generator_.seed(seed);
    shuffle();
    current_ = 0;
  }
};