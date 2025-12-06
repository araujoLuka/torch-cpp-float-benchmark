#pragma once

#include <torch/torch.h>

// Simple CNN for image classification.
// Architecture summary:
//      Conv0: 3 → 8 channels, kernel 3x3, stride 1, padding 1. Output: 8x64x64
//      Pool → 8x32x32
//      Conv1: 8 → 16 channels, 3x3. Output: 16x32x32
//      Pool → 16x16x16
//      Conv2: 16 → 24 channels, 3x3. Output: 24x16x16
//      Pool → 24x8x8
//      Flatten: 1536
//      FC0: 1536 → 256
//      FC1: 256 → num_classes

class Net; // Forward declaration for the module holder

class NetImpl : public torch::nn::Module
{
   public:
    explicit NetImpl(int64_t num_classes);

    torch::Tensor forward(torch::Tensor x);

    // Custom factory: build model and move to desired dtype/device.
    static std::shared_ptr<Net> create(int64_t num_classes, torch::Dtype dtype, torch::Device device);

    // Total number of parameters
    int64_t total_params;

   private:
    void initialize_weights();

    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    // Store only num_classes. Device/dtype are not stored.
    int64_t num_classes;

    // Static architecture constants
    static constexpr int64_t k_c_input = 3;
    static constexpr int64_t k_c_conv1_out = 8;
    static constexpr int64_t k_c_conv2_out = 16;
    static constexpr int64_t k_c_conv3_out = 24;

    static constexpr int64_t k_input_res = 64;
    static constexpr int64_t k_num_pools = 3;
    static constexpr int64_t k_final_spatial = k_input_res >> k_num_pools;  // 64 / 8 = 8

    static constexpr int64_t k_flattened_features = k_c_conv3_out * k_final_spatial * k_final_spatial;
    static constexpr int64_t k_fc1_out_features = 256;
};

TORCH_MODULE(Net);
