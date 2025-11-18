#ifndef NET_HPP
#define NET_HPP

#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <string>

namespace network {

/**
 * @brief Supported data types for the neural network layers.
 */
enum class NetType : uint8_t {
    FLOAT32,
    DOUBLE,
    HALF,
    BFLOAT16,
};

/**
 * @brief Supported devices for running the network.
 */
enum class DeviceType : uint8_t {
    CPU,
    CUDA,
};

/**
 * @brief Configuration structure for defining network parameters.
 */
struct NetConfig {
    std::int64_t input_size;               ///< Size of the input layer
    std::vector<std::int64_t> hidden_sizes;///< Sizes of hidden layers
    std::int64_t output_size;              ///< Size of the output layer
    NetType dtype;                         ///< Desired numeric precision
    DeviceType device;                     ///< Execution device (CPU or CUDA)
};

/**
 * @brief Simple feed-forward neural network with configurable precision and device.
 */
class NetImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new NetImpl object with the given configuration.
     * @param config Network configuration (sizes, precision, device)
     */
    explicit NetImpl(const NetConfig& config);

    /**
     * @brief Forward pass through the network.
     * @param x Input tensor
     * @return Output tensor after forward computation
     */
    torch::Tensor forward(const torch::Tensor& x);

private:
    NetConfig config_;                                     ///< Stored configuration
    torch::nn::ModuleList layers_{nullptr};                ///< Sequential list of fully connected layers
    torch::TensorOptions tensor_options_;                  ///< Internal tensor options (dtype + device)

    /**
     * @brief Helper method to build tensor options based on configuration.
     */
    void configureTensorOptions();

    /**
     * @brief Helper method to create network layers using the configuration.
     */
    void buildLayers();
};

// Register module with LibTorch
TORCH_MODULE(Net);

}  // namespace network

#endif  // NET_HPP
