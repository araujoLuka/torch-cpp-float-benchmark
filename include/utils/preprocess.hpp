#pragma once
#include <torch/torch.h>

struct MetaPreprocess {
    std::vector<double> norm_mean;
    std::vector<double> norm_std;
};

torch::Tensor preprocess_batch(
    torch::Tensor inputs,
    const torch::Device& device,
    torch::ScalarType dtype
);
