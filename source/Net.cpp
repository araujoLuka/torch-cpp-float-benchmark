#include "Net.hpp"

#include <cmath>

// ------------------------------------------------------------
// Constructor (CPU float32 only — factory will move to dtype/device)
// ------------------------------------------------------------
NetImpl::NetImpl(int64_t num_classes) : num_classes(num_classes)
{
    conv1 = register_module(
        "conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(k_c_input, k_c_conv1_out, 3).stride(1).padding(1)));

    conv2 = register_module(
        "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(k_c_conv1_out, k_c_conv2_out, 3).stride(1).padding(1)));

    conv3 = register_module(
        "conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(k_c_conv2_out, k_c_conv3_out, 3).stride(1).padding(1)));

    const int64_t conv_params =
        (k_c_input * k_c_conv1_out * 3 * 3 + k_c_conv1_out) +
        (k_c_conv1_out * k_c_conv2_out * 3 * 3 + k_c_conv2_out) +
        (k_c_conv2_out * k_c_conv3_out * 3 * 3 + k_c_conv3_out);

    pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
    relu = torch::nn::ReLU();

    fc1 = register_module("fc1", torch::nn::Linear(k_flattened_features, k_fc1_out_features));
    fc2 = register_module("fc2", torch::nn::Linear(k_fc1_out_features, num_classes));

    const int64_t fc_params =
        (k_flattened_features * k_fc1_out_features + k_fc1_out_features) +
        (k_fc1_out_features * num_classes + num_classes);

    total_params = conv_params + fc_params;

    initialize_weights();
}

// ------------------------------------------------------------
// Forward
// ------------------------------------------------------------
torch::Tensor NetImpl::forward(torch::Tensor x)
{
    TORCH_CHECK(x.dim() == 4, "Input must be [N,C,H,W]. Got: ", x.sizes());
    TORCH_CHECK(x.size(1) == 3, "Input must have 3 channels.");
    TORCH_CHECK(x.size(2) == k_input_res && x.size(3) == k_input_res, "Input must be 64x64.");

    // ---- Feature extractor ----
    x = relu(conv1(x));
    x = pool(x);

    x = relu(conv2(x));
    x = pool(x);

    x = relu(conv3(x));
    x = pool(x);

    // ---- Flatten ----
    x = x.view({x.size(0), -1});

    // ---- Classifier ----
    x = relu(fc1(x));
    x = fc2(x);

    return x;
}

// ------------------------------------------------------------
// Weight initialization: Kaiming for weights, zeros for bias
// ------------------------------------------------------------
void NetImpl::initialize_weights()
{
    for (auto& pair : named_parameters(/*recurse=*/true))
    {
        auto& p = pair.value();
        if (p.dim() > 1) { torch::nn::init::kaiming_uniform_(p, std::sqrt(5.0)); }
        else {
            torch::nn::init::constant_(p, 0.0);
        }
    }
}

// ------------------------------------------------------------
// Factory: build model → move to dtype/device → return
// ------------------------------------------------------------
std::shared_ptr<Net> NetImpl::create(int64_t num_classes, torch::Dtype dtype, torch::Device device)
{
    auto model = std::make_shared<Net>(num_classes);

    // Important: call module->to(device, dtype)
    model->get()->to(device, dtype, true);

    return model;
}
