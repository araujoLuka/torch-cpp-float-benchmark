#pragma once

#include <torch/torch.h>
#include <vector>
#include <iostream>

// Net.hpp
// Rede CNN simples, parametrizável e com suporte a torch::Dtype genérico.
// Arquitetura exemplo:
//   Conv2d -> ReLU -> MaxPool
//   Conv2d -> ReLU -> MaxPool
//   (opcional Dropout)
//   Flatten -> Linear -> ReLU -> Linear (num_classes)
//
// Importante: após a criação, chamamos this->to(device, dtype) para forçar
// que todos os parâmetros e buffers usem o dtype desejado.

class NetImpl : public torch::nn::Module {
public:
    // módulos
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}; // usamos até 3 convs
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Functional relu{nullptr};
    torch::nn::MaxPool2d pool{nullptr};

    // TensorOptions para criar tensores no dtype correto
    torch::Device device_;
    torch::Dtype dtype_;
    torch::TensorOptions tensor_options_;

    // parâmetros da arquitetura
    int64_t num_classes_;
    bool use_dropout_;

    // Construtor: passe dtype, device e parâmetros de arquitetura
    NetImpl(torch::Dtype dtype = torch::kFloat32,
            torch::Device device = torch::Device("cpu"),
            int64_t in_channels = 3,
            int64_t num_classes = 10,
            bool use_dropout = false,
            int64_t base_filters = 32)
        : device_(device), dtype_(dtype),
          tensor_options_(torch::TensorOptions().device(device_).dtype(dtype_)),
          num_classes_(num_classes),
          use_dropout_(use_dropout)
    {
        // Camada convolucional 1: base_filters filtros, kernel 3x3
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, base_filters, /*kernel_size=*/3).stride(1).padding(1)));
        // Camada convolucional 2: 2*base_filters filtros
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(base_filters, base_filters * 2, 3).stride(1).padding(1)));
        // (Opcional) Camada convolucional 3
        conv3 = register_module("conv3",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(base_filters * 2, base_filters * 4, 3).stride(1).padding(1)));

        // Pooling 2x2 e ReLU funcional
        pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        relu = torch::nn::Functional(torch::relu);

        if (use_dropout_) {
            dropout = register_module("dropout", torch::nn::Dropout(0.5));
        }

        // fc layers (os tamanhos dependem do tamanho da imagem de entrada)
        // Por simplicidade: assumimos imagens redimensionadas para 64x64 antes de entrar na rede
        // => após 3 pools (div por 8): 64/8 = 8 -> feature map 8x8
        const int64_t final_spatial = 8; // documente: ajuste quando usar outra resolução
        int64_t flattened_features = (base_filters * 4) * final_spatial * final_spatial;

        fc1 = register_module("fc1", torch::nn::Linear(flattened_features, 256));
        fc2 = register_module("fc2", torch::nn::Linear(256, num_classes_));

        // Inicialização de pesos (Xavier) e conversão para dtype desejado
        initialize_weights();
        // Garante que todos os parâmetros e buffers sejam convertidos para device+dtype
        this->to(device_, dtype_);
    }

    // Forward pass: espera entrada em formato [N, C, H, W].
    torch::Tensor forward(torch::Tensor x) {
        // Assegura que o input esteja no dtype desejado (importantíssimo para FP16/bfloat16)
        if (x.dtype() != dtype_) {
            x = x.to(dtype_);
        }
        // Convolução 1
        x = relu->forward(conv1->forward(x));
        x = pool->forward(x);

        // Convolução 2
        x = relu->forward(conv2->forward(x));
        x = pool->forward(x);

        // Convolução 3 (opcional)
        x = relu->forward(conv3->forward(x));
        x = pool->forward(x);

        if (use_dropout_) x = dropout->forward(x);

        x = x.view({x.size(0), -1}); // flatten
        x = relu->forward(fc1->forward(x));
        x = fc2->forward(x); // logits; aplicar softmax externamente se quiser

        return x;
    }

    // Utility: inicializa pesos (Xavier) respeitando dtype
    void initialize_weights() {
        for (auto &pair : this->named_parameters(/*recurse=*/true)) {
            auto &name = pair.key();
            auto &param = pair.value();
            if (param.dim() > 1) {
                torch::nn::init::xavier_uniform_(param, /*gain=*/std::sqrt(2.0));
            } else {
                // biases
                torch::nn::init::constant_(param, 0);
            }
        }
    }

    // Método para forçar dtype/device se quiser trocar depois
    void to_dtype(torch::Dtype dtype, torch::Device device = torch::Device("cpu")) {
        dtype_ = dtype;
        device_ = device;
        tensor_options_ = torch::TensorOptions().device(device_).dtype(dtype_);
        this->to(device_, dtype_);
    }
};

// Necessário para registrar com module holder
TORCH_MODULE(Net);
