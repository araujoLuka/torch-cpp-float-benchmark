#pragma once
#ifndef DATASET_MEMMAP_HPP
#define DATASET_MEMMAP_HPP

#include <torch/torch.h>
#include <string>
#include <sys/mman.h>

// High-performance memmap dataset for LibTorch
// - Expects two files:
//    * images memmap: raw contiguous values with layout [N, C, H, W]
//      element type: uint8 (ImageDtype::UINT8) or float32 (ImageDtype::FLOAT32)
//    * labels memmap: N values of uint64_t
// - Zero-copy by default (returns tensors created with from_blob).
// - If you absolutely need worker-safety with num_workers>0 and independent memory,
//   set "force_clone_in_get" true (this will clone per-sample — expensive).

class MemmapTensorDataset : public torch::data::Dataset<MemmapTensorDataset> {
public:
    enum class ImageDtype { UINT8, FLOAT32 };

    // images_path: path to images memmap
    // labels_path: path to labels memmap
    // N,C,H,W: dimensions
    // dtype: element dtype of images file
    // force_clone_in_get: if true, get() will clone returned tensors (useful in some multi-worker setups)
    MemmapTensorDataset(const std::string& images_path,
                       const std::string& labels_path,
                       size_t N,
                       size_t C,
                       size_t H,
                       size_t W,
                       ImageDtype dtype,
                       bool force_clone_in_get = false);

    MemmapTensorDataset(const MemmapTensorDataset&);

    MemmapTensorDataset(MemmapTensorDataset&&) noexcept;
    MemmapTensorDataset& operator=(MemmapTensorDataset&&) noexcept;

    ~MemmapTensorDataset() override;

    // returns a single sample (image tensor [C,H,W], label scalar tensor)
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

    // Accessors
    size_t N() const { return N_; }
    size_t C() const { return C_; }
    size_t H() const { return H_; }
    size_t W() const { return W_; }

private:
    void open_and_map_files();

    std::string images_path_;
    std::string labels_path_;
    size_t N_, C_, H_, W_;
    ImageDtype dtype_;

    int images_fd_{-1};
    int labels_fd_{-1};
    void* images_map_{MAP_FAILED};
    void* labels_map_{MAP_FAILED};
    size_t images_map_bytes_{0};
    size_t labels_map_bytes_{0};

    size_t element_size_bytes_{0};
    size_t sample_bytes_{0};

    bool force_clone_in_get_{false};
};

#endif // DATASET_MEMMAP_HPP

