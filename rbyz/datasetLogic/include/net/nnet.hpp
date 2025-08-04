#pragma once

class NNet : public torch::nn::Module {
 public:
    NNet() = default;
    virtual ~NNet() = default;

    virtual torch::Tensor forward(torch::Tensor x) = 0;
};
