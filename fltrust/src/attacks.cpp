#include "../include/attacks.hpp"

// These functions are translated from https://github.com/encryptogroup/SAFEFL/blob/main/attacks.py

/**
 * No attack is performed.
 * @param v: list of gradients
 * @param net: model
 * @param lr: learning rate
 * @param f: number of malicious clients, where the first f are malicious
 * @param device: device used in training and inference
 * @return: list of gradients after the trim attack
 */
std::vector<torch::Tensor> no_byz(
    const std::vector<torch::Tensor>& v,
    Net net,
    int lr,
    int f,
    torch::Device device
  ) {
    return v;
}

/**
 * @param v: list of gradients
 * @param net: model
 * @param lr: learning rate
 * @param f: number of malicious clients, where the first f are malicious
 * @param device: device used in training and inference
 * @return: list of gradients after the trim attack
 */
std::vector<torch::Tensor> trim_attack(
    const std::vector<torch::Tensor> &v,
    Net net,
    int lr,
    int f,
    torch::Device device)
{
    std::vector<torch::Tensor> v_attack;
    for (const auto &t : v) {
        if (t.dim() == 1)
            // unsqueeze along dim 1 to create a column vector.
            v_attack.push_back(t.unsqueeze(1)); 
        else
            v_attack.push_back(t);
    }

    auto vi_shape = v_attack[0].sizes();
    auto v_tran = torch::cat(v_attack, 1);

    auto max_tuple = torch::max(v_tran, 1, /*keepdim=*/true);
    auto min_tuple = torch::min(v_tran, 1, /*keepdim=*/true);
    auto maximum_dim = std::get<0>(max_tuple);
    auto minimum_dim = std::get<0>(min_tuple);
    auto direction = torch::sign(torch::sum(v_tran, -1, /*keepdim=*/true));
    auto directed_dim = ((direction > 0).to(torch::kFloat32) * minimum_dim +
                         (direction < 0).to(torch::kFloat32) * maximum_dim);

    for (int i = 0; i < f; i++) {
        auto random_12 = (1.0 + torch::rand(vi_shape, device));
        auto cond1 = (direction * directed_dim > 0).to(torch::kFloat32);
        auto cond2 = (direction * directed_dim < 0).to(torch::kFloat32);
        auto factor = cond1.div(random_12) + cond2 * random_12;
        v_attack[i] = directed_dim * factor;
    }

    // Squeeze back to 1D
    for (size_t i = 0; i < v_attack.size(); i++) {
        if (v[i].dim() == 1)
            v_attack[i] = v_attack[i].squeeze(1);
    }

    return v_attack;
}