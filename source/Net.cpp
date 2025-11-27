#include "Net.hpp"

#include <cmath>

NetImpl::NetImpl(int64_t num_classes, torch::Dtype dtype, torch::Device device)
    : device{device},
      dtype{dtype},
      tensor_options{torch::TensorOptions().device(device).dtype(dtype)},
      num_classes{num_classes}
{
    // Fixed RGB input (3 channels) and explicit channel sizes per layer.
    // Target pattern: 3 (input RGB) -> 64 -> 128 -> 256.

    // Convolution 1: 3 -> 64, 3x3 kernel
    this->conv1 = this->register_module(
        "conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(k_input_channels, k_conv1_out_channels, 3).stride(1).padding(1)));

    // Convolution 2: 64 -> 128, 3x3 kernel
    this->conv2 = this->register_module(
        "conv2", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(k_conv1_out_channels, k_conv2_out_channels, 3).stride(1).padding(1)));

    // Convolution 3: 128 -> 256, 3x3 kernel
    this->conv3 = this->register_module(
        "conv3", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(k_conv2_out_channels, k_conv3_out_channels, 3).stride(1).padding(1)));

    // 2x2 max pooling and ReLU activation
    this->pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
    this->relu = torch::nn::Functional(torch::relu);

    // Fully connected layers.
    // Assumes input images are resized to 64x64 before entering the network.
    // After k_num_pools pooling layers (stride 2 each): 64 / (2^3) = 8, so feature map is 8x8.
    static constexpr int64_t k_flattened_features = k_conv3_out_channels * k_final_spatial * k_final_spatial;

    this->fc1 = this->register_module("fc1", torch::nn::Linear(k_flattened_features, 256));
    this->fc2 = this->register_module("fc2", torch::nn::Linear(256, this->num_classes));

    // Weight initialization (Xavier) and conversion to desired device/dtype
    this->initialize_weights();
    this->to(device, dtype);

    // Network ready. COUT total number of parameters.
    int64_t total_params = 0;
    for (const auto& p : this->parameters())
    { total_params += p.numel(); }
    std::cout << "[i] Net initialized with " << total_params << " parameters. Device: " << device.str()
              << ", Dtype: " << dtype << "\n";
}

// Forward pass: input is [N, C, H, W].
torch::Tensor NetImpl::forward(torch::Tensor x)
{
    // Ensure input matches network dtype (important for FP16/bfloat16).
    if (x.dtype() != this->dtype) { x = x.to(this->dtype); }

    // Convolution + ReLU + Pool blocks
    x = this->relu->forward(this->conv1->forward(x));
    x = this->pool->forward(x);

    x = this->relu->forward(this->conv2->forward(x));
    x = this->pool->forward(x);

    x = this->relu->forward(this->conv3->forward(x));
    x = this->pool->forward(x);

    // Flatten to [N, -1]
    x = x.view({x.size(0), -1});

    // Fully connected classifier head
    x = this->relu->forward(this->fc1->forward(x));
    x = this->fc2->forward(x);  // Logits; apply softmax externally if needed.

    return x;
}

// Initialize weights with Xavier initialization, biases to zero.
void NetImpl::initialize_weights()
{
    for (auto& pair : this->named_parameters(/*recurse=*/true))
    {
        auto& param = pair.value();
        if (param.dim() > 1) { torch::nn::init::xavier_uniform_(param, /*gain=*/std::sqrt(2.0)); }
        else {
            torch::nn::init::constant_(param, 0.0);
        }
    }
}

// Change network dtype and device after construction.
void NetImpl::to_dtype(torch::Dtype dtype, torch::Device device)
{
    this->dtype = dtype;
    this->device = device;
    this->tensor_options = torch::TensorOptions().device(this->device).dtype(this->dtype);
    this->to(this->device, this->dtype, true);
}
