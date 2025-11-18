#include "Net.hpp"

namespace network {

/**
 * @brief Converts NetType and DeviceType into torch::TensorOptions.
 */
void NetImpl::configureTensorOptions() {
    // Determine dtype
    torch::Dtype dtype;
    switch (config_.dtype) {
        case NetType::DOUBLE:
            dtype = torch::kFloat64;
            break;
        case NetType::HALF:
            dtype = torch::kFloat16;
            break;
        case NetType::BFLOAT16:
            dtype = torch::kBFloat16;
            break;
        case NetType::FLOAT32:
        default:
            dtype = torch::kFloat32;
            break;
    }

    // Determine device
    torch::Device device =
        (config_.device == DeviceType::CUDA && torch::cuda::is_available()) ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    tensor_options_ = torch::TensorOptions().dtype(dtype).device(device);
}

/**
 * @brief Builds the layers based on the configuration parameters.
 */
void NetImpl::buildLayers() {
    std::vector<std::int64_t> sizes;
    sizes.reserve(config_.hidden_sizes.size() + 2);
    sizes.push_back(config_.input_size);
    sizes.insert(sizes.end(), config_.hidden_sizes.begin(), config_.hidden_sizes.end());
    sizes.push_back(config_.output_size);

    // Create fully connected layers
    for (std::size_t i = 0; i < sizes.size() - 1; ++i) {
        auto layer =
            torch::nn::Linear(torch::nn::LinearOptions(sizes[i], sizes[i + 1]).dtype(tensor_options_.dtype()).device(tensor_options_.device()));
        register_module("fc" + std::to_string(i + 1), layer);
        layers_->push_back(layer);
    }
}

/**
 * @brief Constructor that initializes and builds the network.
 */
NetImpl::NetImpl(const NetConfig& config) : config_(config), layers_(torch::nn::ModuleList()) {
    configureTensorOptions();
    buildLayers();
}

/**
 * @brief Forward pass through all layers with ReLU activations (except last
 * layer).
 */
torch::Tensor NetImpl::forward(const torch::Tensor& x_input) {
    torch::Tensor x = x_input.to(tensor_options_);

    for (std::size_t i = 0; i < layers_->size(); ++i) {
        x = layers_[i]->as<torch::nn::Linear>()->forward(x);
        if (i < layers_->size() - 1) x = torch::relu(x);
    }

    return torch::log_softmax(x, /*dim=*/1);
}

}  // namespace network
