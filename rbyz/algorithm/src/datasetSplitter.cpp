#include "datasetSplitter.hpp"

#include <algorithm>
#include <ctime>
#include <stdexcept>

#include "logger.hpp"

DatasetSplitter::DatasetSplitter(TrainInputParams& t_params, IRegDatasetMngr& mngr,
                                 std::vector<ClientDataRbyz>& clnt_data_vec)
    : n_clients(clnt_data_vec.size()),
      samples_per_chunk(t_params.chunk_size),
      clnt_vd_proportion(t_params.clnt_vd_proportion),
      mngr(mngr),
      clnt_data_vec(clnt_data_vec),
      vd_indices(n_clients),
      prev_indexes_arrangement(n_clients),
      rng(50) {
  // Used to select the VD splits for each client
  for (int i = 0; i < n_clients; i++) {
    prev_indexes_arrangement[i] = i;
  }

  initClientChunkOffsets(t_params.overwrite_poisoned);
  initValidationDatasetPartitions();
}

void DatasetSplitter::initClientChunkOffsets(int overwrite_poisoned) {
  if (clnt_vd_proportion > 0.25) {
    throw std::runtime_error(
        "clnt_vd_proportion must be <= 0.25, max 25%' overwrite of the client dataset");
  }

  chunk_sz_bytes = mngr.data_info.get_sample_size() * samples_per_chunk;
  int total_offsets = clnt_data_vec[0].dataset_size / chunk_sz_bytes;
  int num_offsets = total_offsets * clnt_vd_proportion;
  clnt_chunks.reserve(num_offsets);

  Logger::instance().log("Total offsets: " + std::to_string(total_offsets) +
                         ", datset size: " + std::to_string(clnt_data_vec[0].dataset_size) +
                         ", num_offsets: " + std::to_string(num_offsets) +
                         ", chunk_sz_bytes: " + std::to_string(chunk_sz_bytes) +
                         ", samples per chunk: " + std::to_string(samples_per_chunk) + "\n");

  if (overwrite_poisoned) {
    // (total_offsets - 1) / (num_offsets - 1) ensures the last offset lands exactly at the end of
    // the available range.
    double step = static_cast<double>(total_offsets - 1) / (num_offsets - 1);
    for (size_t i = 0; i < num_offsets; i++) {
      size_t offset_index = static_cast<size_t>(i * step);
      clnt_chunks.push_back(offset_index * chunk_sz_bytes);
    }
  } else {
    // all the client chunks to overwrite are put together at the top
    for (size_t i = 0; i < num_offsets; i++) {
      clnt_chunks.push_back(i * chunk_sz_bytes);
    }
  }

  Logger::instance().log("Client offsets initialized with " + std::to_string(clnt_chunks.size()) +
                         " offsets for " + std::to_string(n_clients) + " clients\n");

  // Number of test samples the server can insert into each client at most
  samples_per_vd_split = num_offsets * samples_per_chunk;
}

void DatasetSplitter::initValidationDatasetPartitions() {
  // Split the server registered data into n_clients VD sections
  size_t vd_size = mngr.data_info.num_samples / n_clients;
  extra_col_numel = vd_size * 0.1;
  extra_col_size = extra_col_numel * mngr.data_info.get_sample_size();

  if (vd_size < samples_per_chunk) {
    throw std::runtime_error(
        "Not enough samples in the server VD with the given clnt_vd_proportion. "
        "Increase the registered dataset size or decrease the clnt_vd_proportion.");
  }

  if (samples_per_vd_split == 0) {
    throw std::runtime_error(
        "samples_per_vd_split cannot be zero, check the clnt_vd_proportion and samples_per_chunk "
        "values.");
  }

  Logger::instance().log("Splitting registered dataset into " + std::to_string(n_clients) +
                         " VDs of size " + std::to_string(samples_per_vd_split) + "\n");

  // Initialize server VD indexes, it contains the dataset sample indices for each client split
  for (int i = 0; i < n_clients; i++) {
    size_t start_idx = i * vd_size;
    size_t end_idx;

    if (i == n_clients - 1) {
      end_idx = mngr.data_info.num_samples;
    } else {
      end_idx = (i + 1) * vd_size;
    }

    std::vector<size_t> indices;
    indices.reserve(samples_per_vd_split);

    int indices_put = 0;
    size_t idx = start_idx;
    while (indices_put < samples_per_vd_split) {
      // If we reach the end of the current VD split, wrap around to the start
      if (idx + samples_per_chunk > end_idx) {
        idx = start_idx;
      }

      indices.push_back(idx);
      idx += samples_per_chunk;
      indices_put++;
    }

    vd_indices[i] = indices;
    Logger::instance().log("Client " + std::to_string(i) +
                           " image index: " + std::to_string(vd_indices[i][0]) +
                           " size: " + std::to_string(vd_indices[i].size()) + "\n");
  }
}

int DatasetSplitter::getSamplesPerChunk() const {
  return samples_per_chunk;
}

uint32_t DatasetSplitter::getChunkSize() const {
  return chunk_sz_bytes;
}

std::vector<int> DatasetSplitter::generateDerangement() {
  if (n_clients < 2) {
    return prev_indexes_arrangement;  // No derangement possible
  }

  std::vector<int> new_arrangement(n_clients);

  do {
    std::shuffle(prev_indexes_arrangement.begin(), prev_indexes_arrangement.end(), rng);
    for (int i = 0; i < n_clients; i++) {
      new_arrangement[i] = prev_indexes_arrangement[i];
    }

    // Simple swap to fix any remaining fixed points
    for (int i = 0; i < n_clients; i++) {
      if (new_arrangement[i] == i) {
        // If we find a fixed point, swap it with any other element
        for (int j = 0; j < n_clients; j++) {
          if (i != j && new_arrangement[j] != j) {
            std::swap(new_arrangement[i], new_arrangement[j]);
            break;
          }
        }
      }
    }
  } while (!isDerangement(new_arrangement));

  prev_indexes_arrangement = new_arrangement;
  return new_arrangement;
}

bool DatasetSplitter::isDerangement(const std::vector<int>& arrangement) {
  for (int i = 0; i < n_clients; i++) {
    if (arrangement[i] == i) {
      return false;
    }
  }
  return true;
}

std::vector<size_t> DatasetSplitter::getServerIndices(int clnt_idx, std::vector<int> derangement) {
  return vd_indices[derangement[clnt_idx]];
}

std::vector<size_t> DatasetSplitter::getServerIndices(int idx) {
  if (idx < 0 || idx >= n_clients) {
    throw std::out_of_range("[getServerIndices] Client index out of range");
  }
  return vd_indices[idx];
}

std::vector<std::vector<size_t>> DatasetSplitter::getExtraColIndices() {
  std::vector<std::vector<size_t>> extra_col_indices(n_clients);

  for (int i = 0; i < n_clients; i++) {
    extra_col_indices[i].reserve(extra_col_numel);
    for (int j = 0; j < extra_col_numel; j++) {
      extra_col_indices[i].push_back(vd_indices[i][j]);
    }
  }

  return extra_col_indices;
}

size_t DatasetSplitter::getExtraColSectionToRenew() {
  size_t res = vd_indices[extra_col_idx][0];
  extra_col_idx = (extra_col_idx + 1) % n_clients;  // Cycle through the extra column sections
  return res;
}

size_t DatasetSplitter::getExtraColumnSize() {
  return extra_col_size;
}

int DatasetSplitter::getExtraColNumSamplesPerSection() {
  return extra_col_numel;
}

std::vector<size_t> DatasetSplitter::getClientChunks(int clnt_idx, float proportion) {
  if (clnt_idx < 0 || clnt_idx >= n_clients) {
    throw std::out_of_range("[getClientChunks] Client index out of range");
  }

  if (proportion < 0.0 || proportion > 1.0) {
    throw std::invalid_argument("[getClientChunks] Proportion must be between 0 and 1");
  }

  if (proportion == 1.0) {
    return clnt_chunks;
  }

  std::shuffle(clnt_chunks.begin(), clnt_chunks.end(), rng);
  size_t num_chunks = std::max(static_cast<size_t>(clnt_chunks.size() * proportion), 1UL);
  std::vector<size_t> clnt_chunks_subset(clnt_chunks.begin(), clnt_chunks.begin() + num_chunks);

  return clnt_chunks_subset;
}
