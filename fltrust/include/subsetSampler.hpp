#include <torch/torch.h>
#include <optional>
#include <numeric>
#include <vector>
#include <cstring> 

class SubsetSampler : public torch::data::samplers::Sampler<std::vector<size_t>> {
    public:
    // Type alias required by the Sampler interface.
    using BatchRequestType = std::vector<size_t>;

    // Constructor: takes a vector of indices.
    explicit SubsetSampler(std::vector<size_t> indices)
        : indices_(std::move(indices)), current_(0) {}

    // Reset the sampler with an optional new size.
    // Providing a default argument so that a call with no parameters is allowed.
    void reset(std::optional<size_t> new_size = std::nullopt) override {
        if (new_size.has_value()) {
            if (new_size.value() < indices_.size()) {
                indices_.resize(new_size.value());
            }
        }
        current_ = 0;
    }

    // Returns the next batch.
    std::optional<BatchRequestType> next(size_t batch_size) override {
        BatchRequestType batch;
        while (batch.size() < batch_size && current_ < indices_.size()) {
            batch.push_back(indices_[current_++]);
        }
        if (batch.empty()) {
            return std::nullopt;
        }
        return batch;
    }

    // Serialize the sampler state.
    void save(torch::serialize::OutputArchive& archive) const override {
        // Convert indices_ to a tensor for serialization.
        torch::Tensor indices_tensor = torch::tensor(
        std::vector<int64_t>(indices_.begin(), indices_.end()), torch::kInt64);
        torch::Tensor current_tensor = torch::tensor(static_cast<int64_t>(current_), torch::kInt64);
        archive.write("indices", indices_tensor);
        archive.write("current", current_tensor);
    }

    // Deserialize the sampler state.
    void load(torch::serialize::InputArchive& archive) override {
        torch::Tensor indices_tensor, current_tensor;
        archive.read("indices", indices_tensor);
        archive.read("current", current_tensor);
        auto numel = indices_tensor.numel();
        std::vector<int64_t> temp(numel);
        std::memcpy(temp.data(), indices_tensor.data_ptr<int64_t>(), numel * sizeof(int64_t));
        indices_.resize(numel);

        for (size_t i = 0; i < numel; ++i) {
            indices_[i] = static_cast<size_t>(temp[i]);
        }
            current_ = static_cast<size_t>(current_tensor.item<int64_t>());
        }

    private: std::vector<size_t> indices_; size_t current_; 

};