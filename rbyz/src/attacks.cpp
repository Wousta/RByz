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
    const std::vector<torch::Tensor> &v,
    Net net,
    int lr,
    int f,
    torch::Device device)
{
    return v;
}

/**
 * Local model poisoning attack against the trimmed mean aggregation rule.
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
    Logger::instance().log("Trim attack\n");
    std::vector<torch::Tensor> v_attack;
    for (const auto &t : v)
    {
        if (t.dim() == 1)
            // unsqueeze along dim 1 to create a column vector.
            v_attack.push_back(t.unsqueeze(1));
        else
            v_attack.push_back(t);
    }

    //auto vi_shape = v_attack[0].sizes();
    std::vector<int64_t> vi_shape_vec(v_attack[0].sizes().begin(), v_attack[0].sizes().end());
    auto v_tran = torch::cat(v_attack, 1);

    auto max_tuple = torch::max(v_tran, 1, /*keepdim=*/true);
    auto min_tuple = torch::min(v_tran, 1, /*keepdim=*/true);
    auto maximum_dim = std::get<0>(max_tuple);
    auto minimum_dim = std::get<0>(min_tuple);
    auto direction = torch::sign(torch::sum(v_tran, -1, /*keepdim=*/true));
    auto directed_dim = ((direction > 0).to(torch::kFloat32) * minimum_dim +
                         (direction < 0).to(torch::kFloat32) * maximum_dim);

    for (int i = 0; i < f; i++)
    {
        auto random_12 = (1.0 + torch::rand(vi_shape_vec, torch::kCPU));
        auto cond1 = (direction * directed_dim > 0).to(torch::kFloat32);
        auto cond2 = (direction * directed_dim < 0).to(torch::kFloat32);
        auto factor = cond1.div(random_12) + cond2 * random_12;
        v_attack[i] = directed_dim * factor;
    }

    // Squeeze back to 1D
    for (size_t i = 0; i < v_attack.size(); i++)
    {
        if (v[i].dim() == 1)
            v_attack[i] = v_attack[i].squeeze(1);
    }

    return v_attack;
}

/**
 * Local model poisoning attack against the krum aggregation rule.
 * @param v: list of gradients
 * @param net: model
 * @param lr: learning rate
 * @param f: number of malicious clients, where the first f are malicious
 * @param device: device used in training and inference
 * @return: list of gradients after the krum attack
 */
std::vector<torch::Tensor> krum_attack(
    const std::vector<torch::Tensor> &v,
    Net net,
    int lr,
    int f,
    torch::Device device)
{
    const float threshold = 1e-5;

    // Copy input vector to allow modification
    std::vector<torch::Tensor> v_attack = v;
    int n = v_attack.size();

    // Get dimension of the first gradient
    int64_t d = v_attack[0].size(0);

    // Create distance matrix
    auto dist = torch::zeros({n, n}, torch::kCPU);
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            auto d_val = torch::norm(lr * v_attack[i] - lr * v_attack[j], 2);
            dist.index_put_({i, j}, d_val);
            dist.index_put_({j, i}, d_val);
        }
    }

    // Extract benign-to-benign distances
    auto dist_benign = dist.index({torch::indexing::Slice(f), torch::indexing::Slice(f)});
    auto dist_benign_sorted = std::get<0>(torch::sort(dist_benign));

    // Calculate min_dist (minimum sum of distances to n-f-1 closest neighbors)
    auto min_dist = torch::min(torch::sum(
        dist_benign_sorted.index({torch::indexing::Slice(), torch::indexing::Slice(0, n - f - 1)}),
        -1));

    // Calculate max distance to origin among benign updates
    std::vector<torch::Tensor> dist_w_re;
    for (int i = f; i < n; i++)
    {
        dist_w_re.push_back(torch::norm(lr * v_attack[i], 2));
    }
    auto max_dist_w_re = torch::max(torch::stack(dist_w_re));

    // Calculate maximum lambda
    auto max_lambda = min_dist / ((n - 2 * f - 1) * torch::sqrt(torch::tensor(static_cast<float>(d)))) +
                      max_dist_w_re / torch::sqrt(torch::tensor(static_cast<float>(d)));

    // Start with max_lambda and iteratively reduce if needed
    float actual_lambda = max_lambda.item<float>();
    auto sorted_dist = std::get<0>(torch::sort(dist, -1));

    // Find the initial update (krum selection before attack)
    auto sums = torch::sum(
        sorted_dist.index({torch::indexing::Slice(), torch::indexing::Slice(0, n - f - 1)}),
        -1);
    auto min_idx = torch::argmin(sums);
    auto update_before = v_attack[min_idx.item<int64_t>()];

    // Iteratively adjust lambda until threshold or convergence
    while (actual_lambda > threshold)
    {
        // Apply attack to first f clients
        for (int i = 0; i < f; i++)
        {
            v_attack[i] = -actual_lambda * torch::sign(update_before);
        }

        // Recompute distances
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                auto d_val = torch::norm(v_attack[i] - v_attack[j], 2);
                dist.index_put_({i, j}, d_val);
                dist.index_put_({j, i}, d_val);
            }
        }

        // Find new global update (krum selection)
        sorted_dist = std::get<0>(torch::sort(dist, -1));
        sums = torch::sum(
            sorted_dist.index({torch::indexing::Slice(), torch::indexing::Slice(0, n - f - 1)}),
            -1);
        min_idx = torch::argmin(sums);

        // Break if the attack succeeded (malicious client selected)
        if (min_idx.item<int64_t>() < f)
        {
            break;
        }
        else
        {
            // Otherwise reduce lambda and try again
            actual_lambda /= 2;
        }
    }

    return v_attack;
}
