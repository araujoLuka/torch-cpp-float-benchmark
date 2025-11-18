#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <random>
#include "Net.hpp"

// Exemplo de dataset mínimo que carrega arquivos .pt (cada arquivo contém um tensor de imagem e um label)
// Formato sugerido por você: pré-processar PKLot e Fruits360 com Python e salvar cada amostra como um dict:
//   sample = {"image": image_tensor, "label": torch.tensor(label, dtype=torch.int64)}
//   torch.save(sample, "path/to/sample_00001.pt")
//
// O PtDataset abaixo carrega uma lista de paths e, no get(), usa torch::load para recuperar o tensor.
// Essa abordagem evita dependências C++ extras e mantém controle experimental.

class PtDataset : public torch::data::datasets::Dataset<PtDataset> {
private:
    std::vector<std::string> files_;
public:
    explicit PtDataset(const std::vector<std::string>& files) : files_(files) {}

    // Retorna um sample (tensor, label)
    torch::data::Example<> get(size_t index) override {
        const auto &path = files_.at(index);
        // Carrega um mapa salvo: {"image": Tensor, "label": Tensor}
        std::map<std::string, torch::Tensor> sample;
        torch::load(sample, path); // requer que Python saved dict uses same keys
        torch::Tensor image = sample["image"];
        torch::Tensor label = sample["label"].to(torch::kInt64);
        return {image, label};
    }

    torch::optional<size_t> size() const override {
        return files_.size();
    }
};

// Função utilitária para ler um arquivo de paths (cada linha = caminho para .pt)
std::vector<std::string> read_paths_file(const std::string &path_list) {
    std::vector<std::string> out;
    std::ifstream in(path_list);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

// Calcula acurácia simples (top-1)
double compute_accuracy(torch::Tensor outputs, torch::Tensor labels) {
    // outputs: [N, C] logits; labels: [N]
    auto preds = outputs.argmax(1);
    auto correct = preds.eq(labels).to(torch::kInt64).sum().item<int64_t>();
    auto total = labels.size(0);
    return static_cast<double>(correct) / static_cast<double>(total);
}

int main(int argc, char** argv) {
    // Parâmetros (poderia usar cxxopts/argparse; aqui simplificamos)
    std::string train_list = "train_paths.txt";
    std::string val_list = "val_paths.txt";
    std::string test_list = "test_paths.txt";
    int64_t epochs = 10;
    int64_t batch_size = 32;
    double lr = 1e-3;
    int64_t seed = 42;
    std::string dtype_arg = "float32"; // opções: float64, float32, float16, bfloat16
    int64_t num_classes = 10;
    bool use_dropout = false;

    // (Opcional) parse argv for minimal overrides
    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--dtype" && i+1<argc) dtype_arg = argv[++i];
        if (a=="--epochs" && i+1<argc) epochs = std::stoll(argv[++i]);
        if (a=="--batch_size" && i+1<argc) batch_size = std::stoll(argv[++i]);
        if (a=="--train" && i+1<argc) train_list = argv[++i];
        if (a=="--val" && i+1<argc) val_list = argv[++i];
        if (a=="--test" && i+1<argc) test_list = argv[++i];
        if (a=="--classes" && i+1<argc) num_classes = std::stoll(argv[++i]);
        if (a=="--dropout") use_dropout = true;
    }

    // Mapeia string para torch::Dtype
    torch::Dtype dtype = torch::kFloat32;
    if (dtype_arg == "float64") dtype = torch::kFloat64;
    else if (dtype_arg == "float32") dtype = torch::kFloat32;
    else if (dtype_arg == "float16") dtype = torch::kFloat16;
    else if (dtype_arg == "bfloat16") dtype = torch::kBFloat16;
    else {
        std::cerr << "dtype not recognized, defaulting to float32\n";
    }

    // Seed / Reprodutibilidade
    torch::manual_seed(seed);
    std::srand(static_cast<unsigned>(seed));
    // Controla o número de threads BLAS/OMP para reduzir variabilidade entre execuções
    torch::set_num_threads(1);

    // Device (apenas CPU neste experimento)
    torch::Device device("cpu");

    // Carrega paths dos datasets (arquivos .txt com lista de .pt)
    auto train_paths = read_paths_file(train_list);
    auto val_paths = read_paths_file(val_list);
    auto test_paths = read_paths_file(test_list);

    // Cria datasets
    auto train_dataset = PtDataset(train_paths)
            .map(torch::data::transforms::Stack<>()); // empilha para batches

    auto val_dataset = PtDataset(val_paths)
            .map(torch::data::transforms::Stack<>());

    auto test_dataset = PtDataset(test_paths)
            .map(torch::data::transforms::Stack<>());

    // DataLoaders
    auto train_loader = torch::data::make_data_loader(std::move(train_dataset),
                        torch::data::DataLoaderOptions().batch_size(batch_size).workers(0));
    auto val_loader = torch::data::make_data_loader(std::move(val_dataset),
                        torch::data::DataLoaderOptions().batch_size(batch_size).workers(0));
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset),
                        torch::data::DataLoaderOptions().batch_size(batch_size).workers(0));

    // Construção do modelo
    Net model(dtype, device, /*in_channels=*/3, /*num_classes=*/num_classes, use_dropout, /*base_filters=*/32);
    model->train(); // modo treino por padrão

    // Otimizador
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));

    // Critério de perda (CrossEntropy espera logits e labels int64)
    // NOTA: loss opera com tensores convertidos automaticamente; certifique-se que labels sejam int64

    // Logs / medição de tempo
    auto t_start = std::chrono::steady_clock::now();
    std::cout << "Training started. dtype=" << dtype_arg << " epochs=" << epochs << " batch_size=" << batch_size << "\n";

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        auto epoch_start = std::chrono::steady_clock::now();
        double epoch_loss = 0.0;
        int64_t batch_idx = 0;
        double epoch_acc = 0.0;
        int64_t num_samples = 0;

        // ======== TREINO ========
        model->train();
        for (auto& batch : *train_loader) {
            // batch.data: tensor [B, C, H, W]
            auto inputs = batch.data.to(device);
            auto targets = batch.target.to(device, torch::kInt64);

            // Assegura dtype dos inputs (muito importante para FP16 / BF16)
            if (inputs.dtype() != dtype) inputs = inputs.to(dtype);

            optimizer.zero_grad();
            auto outputs = model->forward(inputs);
            auto loss = torch::nn::functional::cross_entropy(outputs, targets);
            loss.backward();
            optimizer.step();

            // Acumula estatísticas
            double loss_scalar = loss.item<double>();
            epoch_loss += loss_scalar * inputs.size(0);
            epoch_acc += compute_accuracy(outputs.detach().to(torch::kCPU), targets.to(torch::kCPU)) * inputs.size(0);
            num_samples += inputs.size(0);
            batch_idx++;
        }

        epoch_loss /= static_cast<double>(num_samples);
        epoch_acc /= static_cast<double>(num_samples);

        // ======== VALIDAÇÃO ========
        model->eval();
        double val_loss = 0.0;
        double val_acc = 0.0;
        int64_t val_samples = 0;
        for (auto& batch : *val_loader) {
            auto inputs = batch.data.to(device);
            auto targets = batch.target.to(device, torch::kInt64);
            if (inputs.dtype() != dtype) inputs = inputs.to(dtype);

            // No grad during validation
            torch::NoGradGuard no_grad;
            auto outputs = model->forward(inputs);
            auto loss = torch::nn::functional::cross_entropy(outputs, targets);

            val_loss += loss.item<double>() * inputs.size(0);
            val_acc += compute_accuracy(outputs.to(torch::kCPU), targets.to(torch::kCPU)) * inputs.size(0);
            val_samples += inputs.size(0);
        }
        if (val_samples > 0) {
            val_loss /= static_cast<double>(val_samples);
            val_acc /= static_cast<double>(val_samples);
        }

        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count();

        // Log simples
        std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                  << "train_loss=" << epoch_loss << " train_acc=" << epoch_acc
                  << " val_loss=" << val_loss << " val_acc=" << val_acc
                  << " time_ms=" << epoch_time << "\n";

        // Placeholder LIKWID (ex.: start/stop region) - futuramente você integrará aqui comandos/shell calls
        // std::cout << "[LIKWID] region stop/start placeholder\n";
    }

    auto t_end = std::chrono::steady_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "Training finished. Total time ms = " << total_time_ms << "\n";

    // ======== TESTE FINAL ========
    model->eval();
    double test_loss = 0.0;
    double test_acc = 0.0;
    int64_t test_samples = 0;
    for (auto& batch : *test_loader) {
        auto inputs = batch.data.to(device);
        auto targets = batch.target.to(device, torch::kInt64);
        if (inputs.dtype() != dtype) inputs = inputs.to(dtype);

        torch::NoGradGuard no_grad;
        auto outputs = model->forward(inputs);
        auto loss = torch::nn::functional::cross_entropy(outputs, targets);
        test_loss += loss.item<double>() * inputs.size(0);
        test_acc += compute_accuracy(outputs.to(torch::kCPU), targets.to(torch::kCPU)) * inputs.size(0);
        test_samples += inputs.size(0);
    }
    if (test_samples>0) {
        test_loss /= static_cast<double>(test_samples);
        test_acc /= static_cast<double>(test_samples);
    }

    std::cout << "Test loss=" << test_loss << " test_acc=" << test_acc << "\n";

    // Salva modelo (opcional). Para salvar um scriptmodule (TorchScript) converteria para script,
    // mas para fins de reuso científico, pode salvar o state_dict:
    torch::save(model, "model_full.pt"); // salva o module completo (atenção ao dtype)
    torch::save(model->state_dict(), "model_state_dict.pt");

    return 0;
}
