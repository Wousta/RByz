#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "entities.hpp"
#include "manager/iRegDatasetMngr.hpp"

/**
 * @brief Class to split the registered dataset into n_clients.
 * During RByz, the server will insert data from its registered dataset into the clients' datasets.
 * The server will split the registered dataset into n_clients Validation Datasets (VD),
 * and each client will receive data from a different Validation Dataset in each round.
 */
class DatasetSplitter {
 private:
  const int n_clients;
  const int samples_per_chunk;
  const float clnt_vd_proportion;  // Proportion of the client dataset that will be overwritten with
                                   // server data
  int samples_per_vd_split = 0;
  int extra_col_numel;  // Number of samples in the "extra" column for VD, if used
  size_t extra_col_size;
  int extra_col_idx = 0;
  uint32_t chunk_sz_bytes;
  IRegDatasetMngr& mngr;
  std::vector<ClientDataRbyz>& clnt_data_vec;
  std::vector<std::vector<size_t>> vd_indices;  // Vector of indices for the start of each VD
  std::vector<size_t> clnt_chunks;              // Vector of offsets for the clients

  // Vector of integers corresponding to the indices in vd_indices and lbl_offsets that each client
  // will use, indices cannot be repeated in consecutive rounds per client
  std::vector<int> prev_indexes_arrangement;
  std::mt19937 rng;

  /**
   * @brief Initializes client chunk offsets for test data injection.
   *
   * Calculates and distributes chunk offsets evenly across the client dataset based
   * on the clnt_vd_proportion passed. These offsets determine where server data will be injected
   * into client datasets during RByz operations. The offsets are distributed uniformly
   * to ensure even coverage across the client's data space.
   *
   * @param overwrite_poisoned If true, the server will overwrite poisoned samples in the client
   * dataset.
   * @throws std::runtime_error if clnt_vd_proportion exceeds 0.25 (25% limit)
   * @return The number of samples in total to be inserted to each client.
   */
  void initClientChunkOffsets(int overwrite_poisoned);

  /**
   * @brief Initializes validation dataset (VD) partitions.
   *
   * Divides the server's registered dataset into n_clients equal sections and creates
   * sampling indices for each partition. Each VD contains indices that start at regular
   * intervals (samples_per_chunk) within its assigned section. If the samples per VD split
   * exceeds the number of samples in the section, it wraps around to the start of the section
   * This may cause to write a lot of repeated samples to the clients if the clnt_vd_proportion is
   * high But this can be mitigated by renewing the server VD dataset often.
   *
   * The extra column is divided inbetween VD partitions, each piece of the extra column starts at
   * the beginning of each VD partition, and it has size dependent on n_clients and full VD_size.
   *
   * The last client receives any remaining samples to handle cases where the dataset
   * size is not perfectly divisible by n_clients.
   */
  void initValidationDatasetPartitions();

 public:
  DatasetSplitter(TrainInputParams& t_params, IRegDatasetMngr& mngr,
                  std::vector<ClientDataRbyz>& clnt_data_vec);

  int getSamplesPerChunk() const;
  uint32_t getChunkSize() const;

  /**
   * Generates a derangement of the previous offset arrangement.
   * A derangement is a permutation where no element appears in its original position.
   * Time Complexity: O(n) where n is the number of clients
   *
   * @return A vector containing the new arrangement of indices for the offset vectors
   *         where no client receives the same partition index as in the previous round
   */
  std::vector<int> generateDerangement();
  bool isDerangement(const std::vector<int>& arrangement);

  /**
   * @brief Gets the server indices for a given client index based on the derangement.
   * This function retrieves the indices of the server's validation dataset (VD) that will be sent
   * to a specific client.
   * @param clnt_idx The index of the client for which to get the server indices.
   * @param derangement The derangement vector that determines the current arrangement of VD
   * samples.
   * @return A vector containing the selected server indices for the client.
   */
  std::vector<size_t> getServerIndices(int clnt_idx, std::vector<int> derangement);
  std::vector<size_t> getServerIndices(int idx);
  std::vector<std::vector<size_t>> getExtraColIndices();
  size_t getExtraColSectionToRenew();
  size_t getExtraColumnSize();
  int getExtraColNumSamplesPerSection();

  /**
   * @brief Gets the client chunks for a given client index. I proportion < 1.0, it returns a random
   * subset of the chunks.
   * @param clnt_idx The index of the client for which to get the chunks.
   * @param proportion The proportion of chunks to return (default is 1.0, meaning all chunks).
   * @return A vector containing the selected chunks for the client.
   */
  std::vector<size_t> getClientChunks(int clnt_idx, float proportion = 1.0);
};