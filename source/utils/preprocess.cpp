#include "utils/preprocess.hpp"

#include <array>

torch::Tensor preprocess_batch(torch::Tensor inputs, const torch::Device& device, torch::ScalarType dtype)
{
    // Default normalization for pretrained PyTorch models (ImageNet)
    static constexpr std::array<float, 3> kNormMean = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> kNormStd = {0.229f, 0.224f, 0.225f};

    static const torch::Tensor mean_global =
        torch::tensor(torch::ArrayRef<float>(kNormMean.data(), kNormMean.size())).view({1, 3, 1, 1});
    static const torch::Tensor std_global =
        torch::tensor(torch::ArrayRef<float>(kNormStd.data(), kNormStd.size())).view({1, 3, 1, 1});
    bool is_uint8 = (inputs.dtype() == torch::kUInt8);

    // 1. Convert uint8 -> float32 and scale to [0,1] only if necessary
    if (is_uint8)
    {
        inputs = inputs.to(torch::kFloat32, /*non_blocking=*/true);
        inputs.div_(255.0f);
    }

    // 2. Always normalize using default pretrained model statistics
    auto mean = mean_global.to(device, inputs.dtype());
    auto std = std_global.to(device, inputs.dtype());

    inputs.sub_(mean).div_(std);

    // 3. Convert to final dtype only if necessary
    if (inputs.dtype() != dtype) { inputs = inputs.to(dtype); }

    return inputs;
}
