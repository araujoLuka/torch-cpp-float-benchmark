#ifndef NET_HPP
#define NET_HPP

#include <torch/torch.h>

#include <vector>

namespace network {

// Alias to simplify Tensor usage
using Tensor = torch::Tensor;

// Possible types for the Network
enum class NetType {
    FLOAT32,
    DOUBLE,
    HALF,
    BFLOAT16,
};

class NetImpl : public torch::nn::Module {
   public:
    NetImpl() = default;
    virtual ~NetImpl() = default;

    const Tensor forward(const Tensor& x_input) {
        Tensor x = x_input;
        x = torch::relu(this->conv1->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(this->conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(this->conv3->forward(x));
        x = torch::max_pool2d(x, 2);

        x = x.view({x.size(0), -1});  // Flatten before fully connected layer
        x = this->fc1->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }

    virtual void setup_layers(void) = 0;

   protected:
    std::vector<NetType> layer_types;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr};

   private:
};

// Register the module with LibTorch
TORCH_MODULE(Net);

}  // namespace network

#endif  // NET_HPP
