#pragma once
#include <torch/torch.h>
#include "InMemoryTensorDataset.hpp"
#include "StreamingMemmap.hpp"

// Dataset que expõe um chunk por vez, carregado a partir de StreamingMemmap.
// Cada chunk vira um InMemoryTensorDataset normal.
// O DataLoader trata como dataset comum.
class StreamingDataset : public torch::data::datasets::Dataset<StreamingDataset>
{
    std::unique_ptr<StreamingMemmap> stream_;
    InMemoryTensorDataset current_ = InMemoryTensorDataset(torch::Tensor(), torch::Tensor());
    bool valid_ = false;

    size_t chunk_size_;
    size_t offset_ = 0;
    size_t total_ = 0;

public:
    StreamingDataset(const std::string& images_path,
                     const std::string& labels_path,
                     size_t N,
                     size_t C,
                     size_t H,
                     size_t W,
                     size_t chunk_size,
                     bool images_are_float32 = false)
        : stream_(std::make_unique<StreamingMemmap>(images_path, labels_path, N, C, H, W, images_are_float32)),
          chunk_size_(chunk_size),
          offset_(0),
          total_(N)
    {}

    StreamingDataset(const StreamingDataset& other) 
    {
        stream_ = std::make_unique<StreamingMemmap>(*(other.stream_));
        chunk_size_ = other.chunk_size_;
        offset_ = other.offset_;
        total_ = other.total_;
        current_ = other.current_;
        valid_ = other.valid_;
    }

    // Carrega o primeiro chunk
    void load_initial()
    {
        offset_ = 0;
        if (offset_ >= total_) {
            valid_ = false;
            return;
        }

        size_t n = std::min(chunk_size_, total_ - offset_);

        torch::Tensor images, labels;
        stream_->load_chunk(offset_, n, images, labels);
        current_ = InMemoryTensorDataset(images, labels);
        valid_ = true;
        offset_ += n;
    }

    // Carrega o próximo chunk sequencialmente
    bool load_next_chunk()
    {
        if (offset_ >= total_) {
            valid_ = false;
            return false;
        }

        size_t n = std::min(chunk_size_, total_ - offset_);

        torch::Tensor images, labels;
        stream_->load_chunk(offset_, n, images, labels);
        current_ = InMemoryTensorDataset(images, labels);
        valid_ = true;
        offset_ += n;

        return true;
    }

    torch::optional<size_t> size() const override
    {
        if (!valid_) return 0;
        return current_.size();
    }

    torch::data::Example<> get(size_t idx) override
    {
        return current_.get(idx);
    }
};
