#pragma once

#include <torch/torch.h>

// Simple CNN for image classification with configurable dtype and device.
// Example architecture:
//   Conv2d -> ReLU -> MaxPool
//   Conv2d -> ReLU -> MaxPool
//   Conv2d -> ReLU -> MaxPool
//   (optional Dropout)
//   Flatten -> Linear -> ReLU -> Linear (num_classes)

class NetImpl : public torch::nn::Module
{
   public:
    NetImpl(int64_t num_classes, torch::Dtype dtype = torch::kFloat32,
            torch::Device device = torch::Device(k_default_device));

    // Forward pass: expects input in [N, C, H, W] format.
    torch::Tensor forward(torch::Tensor x);

    // Change network dtype and device after construction.
    void to_dtype(torch::Dtype dtype, torch::Device device = torch::Device(k_default_device));

   private:
    // Fixed architectural constants
    static constexpr char k_default_device[] = "cpu";
    static constexpr int64_t k_input_channels{3};
    static constexpr int64_t k_conv1_out_channels{64};
    static constexpr int64_t k_conv2_out_channels{128};
    static constexpr int64_t k_conv3_out_channels{256};
    static constexpr int64_t k_input_image_size{64};
    static constexpr int64_t k_num_pools{3};
    static constexpr int64_t k_final_spatial{k_input_image_size / (1 << k_num_pools)};

    // Initialize weights with Xavier initialization, respecting current dtype.
    void initialize_weights();

    // Convolutional, fully-connected, activation and pooling modules
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Functional relu{nullptr};
    torch::nn::MaxPool2d pool{nullptr};

    // Tensor options to consistently create tensors in the right device/dtype
    torch::Device device;
    torch::Dtype dtype;
    torch::TensorOptions tensor_options;

    // Architecture hyperparameters
    int64_t num_classes;
};

// Register NetImpl with torch::nn::ModuleHolder.
TORCH_MODULE(Net);
