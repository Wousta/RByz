#include "../include/attacks.hpp"
#include "global/globalConstants.hpp"

// These functions are translated from
// https://github.com/encryptogroup/SAFEFL/blob/main/attacks.py

/**
 * No attack is performed.
 * @param v: list of gradients
 * @param lr: learning rate
 * @param f: number of malicious clients, where the first f are malicious
 * @param device: device used in training and inference
 * @return: list of gradients after the trim attack
 */
std::vector<torch::Tensor> no_byz(const std::vector<torch::Tensor> &v, int lr,
                                  int f, torch::Device device) {
  return v;
}

/**
 * Local model poisoning attack against the trimmed mean aggregation rule.
 * @param v: list of gradients
 * @param lr: learning rate
 * @param f: number of malicious clients, where the first f are malicious
 * @param device: device used in training and inference
 * @return: list of gradients after the trim attack
 */
std::vector<torch::Tensor> trim_attack(
    const std::vector<torch::Tensor> &v,
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

  for (int i = 0; i < f; i++) {
    auto random_12 = (1.0 + torch::rand(vi_shape_vec, torch::kCPU));
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

/**
 * Local model poisoning attack against the krum aggregation rule.
 * @param v: list of gradients
 * @param lr: learning rate
 * @param f: number of malicious clients, where the first f are malicious
 * @param device: device used in training and inference
 * @return: list of gradients after the krum attack
 */
std::vector<torch::Tensor> krum_attack(
    const std::vector<torch::Tensor> &v,
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


/**
 * Executes a label flip attack based on the specified parameters.
 * 
 * This function implements various label flipping attack strategies against federated learning systems.
 * It supports both random and targeted label flipping attacks with different intensity settings.
 * 
 * @param use_mnist: true if using MNIST dataset, false for CIFAR-10
 * @param t_params: training input parameters containing attack settings:
 *                  - label_flip_type: type of attack to execute (see attack types below)
 *                  - flip_ratio: proportion of labels to flip (0.0 to 1.0)
 * @param mngr: dataset manager interface to perform label flipping operations
 * 
 * Attack Types:
 * - NO_ATTACK (0): No label flipping is performed
 * - RANDOM_FLIP (1): Random label flipping attack where labels are randomly reassigned
 * - TARGETED_FLIP_1 (2): Targeted attack setting 1 - High confusion pairs
 *   * MNIST: 8 → 0 (source class frequently misclassified as target in clean training)
 *   * CIFAR-10: 5 (dog) → 3 (cat) (source class frequently misclassified as target)
 * - TARGETED_FLIP_2 (3): Targeted attack setting 2 - Low confusion pairs  
 *   * MNIST: 1 → 5 (source class infrequently misclassified as target in clean training)
 *   * CIFAR-10: 0 (airplane) → 2 (bird) (source class infrequently misclassified as target)
 * - TARGETED_FLIP_3 (4): Targeted attack setting 3 - Medium confusion pairs
 *   * MNIST: 4 → 9 (moderate confusion between source and target classes)
 *   * CIFAR-10: 1 (automobile) → 9 (truck) (moderate confusion between source and target)
 *
 * Sources for labels chosen: CIFAR-10 -> https://arxiv.org/pdf/2007.08432 | MNIST -> https://arxiv.org/pdf/2407.07818v1
 * 
 * The targeted flip attacks represent three diverse attack conditions:
 * 1. High natural confusion: source ->target pairs where misclassification frequently occurs naturally
 * 2. Low natural confusion: source -> target pairs where misclassification rarely occurs naturally  
 * 3. Medium natural confusion: source -> target pairs with moderate natural misclassification rates
 * 
 * These settings allow evaluation of attack effectiveness under different levels of natural
 * class similarity, providing insights into adversarial robustness across various scenarios.
 * 
 * @throws std::runtime_error if an unknown label_flip_type is provided
 * 
 * @note Uses fixed random seed (42) for reproducible attack execution
 * @note Only affects malicious/Byzantine clients as determined by the federated learning setup
 */
void data_poison_attack(bool use_mnist, TrainInputParams &t_params,
                        IRegDatasetMngr &mngr) {
  std::mt19937 rng(42); // Fixed seed for reproducibility
  int label_flip_type = t_params.label_flip_type;
  float flip_ratio = t_params.flip_ratio;

  if (flip_ratio < 0.0f || flip_ratio > 1.0f) {
    throw std::invalid_argument("Invalid flip_ratio. Must be in [0.0, 1.0]");
  }

  // Define target mappings: [setting][dataset][source, target]
  // dataset: 0=MNIST, 1=CIFAR-10
  const int target_mappings[3][2][2] = {
      {{8, 0}, {5, 3}}, // TARGETED_FLIP_1
      {{1, 5}, {0, 2}}, // TARGETED_FLIP_2
      {{4, 9}, {1, 9}}  // TARGETED_FLIP_3
  };

  switch (label_flip_type) {
  case NO_ATTACK:
    break;
  case RANDOM_FLIP:
    mngr.flipLabelsRandom(flip_ratio, rng);
    break;
  case CORRUPT_IMAGES_RNG:
    mngr.corruptImagesRandom(flip_ratio, rng);
    break;
  case TARGETED_FLIP_1:
  case TARGETED_FLIP_2:
  case TARGETED_FLIP_3: {
    int setting = label_flip_type - TARGETED_FLIP_1; // Convert to 0-based index
    int dataset = use_mnist ? 0 : 1;
    int source = target_mappings[setting][dataset][0];
    int target = target_mappings[setting][dataset][1];

    mngr.flipLabelsTargeted(source, target, flip_ratio, rng);
    break;
  }
  case TARGETED_FLIP_4: {
    // Attack done in FLtrust paper
    for (int i = 0; i < NUM_CLASSES; i++) {
        int src = i;
        int target = NUM_CLASSES - src - 1;
        mngr.flipLabelsTargeted(src, target, 1.0f, rng);
    }
    break;
  }
  default:
    throw std::runtime_error("Unknown label flip type");
  }
}

void progressiveVDColAttack(float prop, IRegDatasetMngr &mngr, std::vector<std::vector<size_t>> &extra_col_indices) {
  int minority = mngr.n_clients / 2;

  if (mngr.n_clients % 2 == 0)
    minority -= 1;

  for (int i = 0; i < minority; i++) {
    auto &extra_indices = extra_col_indices[i];
    int numel_corrupted = static_cast<int>(prop * extra_indices.size());
    if (extra_indices.empty()) {
      continue; // No extra columns to corrupt for this client
    }

    for (int j = 0; j < numel_corrupted; j++) {
      int idx = extra_indices[j];
      mngr.corruptImage(idx);
    }
  }
}