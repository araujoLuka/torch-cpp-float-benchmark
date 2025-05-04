/**
 * @file Net.hpp
 * @brief Header file for the neural network class.
 * @details This file defines a simple convolutional neural network (CNN) using LibTorch.
 *          The network consists of three convolutional layers followed by a fully connected layer.
 *          It is designed to be used with float, double, or c10::Half data types.
 */
#ifndef TORCH_CPP_FLOAT_BENCHMARK_NET_HPP
#define TORCH_CPP_FLOAT_BENCHMARK_NET_HPP

#include <torch/torch.h>
#include <type_traits>
#include <memory>

namespace network {

// Alias to simplify type declarations
template<typename T>
using Tensor = torch::Tensor;

// Generic neural network class
template <typename scalar_t>
class NetImpl : public torch::nn::Module {
    static_assert(
        std::is_same_v<scalar_t, float> ||
        std::is_same_v<scalar_t, double> ||
        std::is_same_v<scalar_t, c10::Half>,
        "scalar_t must be float, double, or c10::Half"
    );

public:
    NetImpl()
        : conv1(torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)),
          conv2(torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1)),
          conv3(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
          fc1(64 * 4 * 4, 2) // Adjust if input image size changes
    {
        this->register_module("conv1", this->conv1);
        this->register_module("conv2", this->conv2);
        this->register_module("conv3", this->conv3);
        this->register_module("fc1", this->fc1);
    }

    // Forward method (input tensor passed as const reference)
    [[nodiscard]]
    Tensor<scalar_t> forward(const Tensor<scalar_t>& x_input) noexcept {
        Tensor<scalar_t> x = x_input;
        x = torch::relu(this->conv1->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(this->conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(this->conv3->forward(x));
        x = torch::max_pool2d(x, 2);

        x = x.view({x.size(0), -1}); // Flatten before fully connected layer
        x = this->fc1->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }

    // Default destructor (smart pointers handle memory)
    ~NetImpl() = default;

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr};

    // Future extension: custom layer can be added here
    // std::shared_ptr<CustomLayer<scalar_t>> custom_layer{nullptr};
};

// Macro required to register the module with LibTorch
template <typename scalar_t>
TORCH_MODULE_TEMPLATE(Net, scalar_t);

} // namespace network

#endif // TORCH_CPP_FLOAT_BENCHMARK_NET_HPP
