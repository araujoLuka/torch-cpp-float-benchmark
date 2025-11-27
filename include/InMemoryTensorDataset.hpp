#pragma once

#include <torch/torch.h>
#include <utility>

class InMemoryTensorDataset : public torch::data::datasets::Dataset<InMemoryTensorDataset>
{
    torch::Tensor images_;  // [N, C, H, W], uint8 or float
    torch::Tensor labels_;  // [N], int64
   public:
    inline InMemoryTensorDataset(torch::Tensor images, torch::Tensor labels, torch::Tensor indices = {})
        : images_(std::move(images)), labels_(std::move(labels))
    {
    }

    inline torch::data::Example<> get(size_t idx) override
    {
        auto img = images_[idx];  // [C,H,W]
        auto lbl = labels_[idx];  // scalar int64 tensor
        return {img, lbl};
    }

    inline torch::optional<size_t> size() const override
    {
        return static_cast<size_t>(labels_.size(0));
    }
};

