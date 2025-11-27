#include "utils/preprocess.hpp"

#include <unordered_map>

torch::Tensor preprocess_batch(torch::Tensor inputs, const torch::Device& device, torch::ScalarType dtype,
                               const MetaPreprocess& meta)
{
    static bool norm_initialized = false;
    // Mean/std na CPU
    static torch::Tensor mean_cpu = torch::tensor({0.0f, 0.0f, 0.0f}).view({1, 3, 1, 1});
    static torch::Tensor std_cpu = torch::tensor({1.0f, 1.0f, 1.0f}).view({1, 3, 1, 1});
    // Cache device-specific
    static std::unordered_map<void*, torch::Tensor> mean_cache;
    static std::unordered_map<void*, torch::Tensor> std_cache;

    bool needs_normalize = (meta.norm_mean.size() == 3 && meta.norm_std.size() == 3);
    bool is_uint8 = (inputs.dtype() == torch::kUInt8);

    // 1. Converte uint8 -> float32 e escala para [0,1] somente se necessário
    if (is_uint8)
    {
        inputs = inputs.to(torch::kFloat32, /*non_blocking=*/true);
        inputs.div_(255.0f);
    }

    // 2. Normalização apenas se mean/std existirem
    if (needs_normalize)
    {
        void* dev_key = device.is_cpu()       ? nullptr
                        : device.index() >= 0 ? reinterpret_cast<void*>(static_cast<intptr_t>(device.index()))
                                              : reinterpret_cast<void*>(-1);

        if (!norm_initialized)
        {
            norm_initialized = true;
            mean_cpu = torch::tensor({(float)meta.norm_mean[0], (float)meta.norm_mean[1], (float)meta.norm_mean[2]})
                           .view({1, 3, 1, 1});
            std_cpu = torch::tensor({(float)meta.norm_std[0], (float)meta.norm_std[1], (float)meta.norm_std[2]})
                          .view({1, 3, 1, 1});

            if (!mean_cache.count(dev_key))
            {
                mean_cache[dev_key] = mean_cpu.to(device, inputs.dtype());
                std_cache[dev_key] = std_cpu.to(device, inputs.dtype());
            }
        }
        else {
            auto& mean = mean_cache[dev_key];
            auto& std = std_cache[dev_key];

            inputs.sub_(mean).div_(std);
        }
    }

    // 3. Converte para dtype final apenas se necessário
    if (inputs.dtype() != dtype) { inputs = inputs.to(dtype); }

    return inputs;
}
