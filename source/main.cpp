#include <ATen/core/Dict.h>
#include <torch/torch.h>
#include <torch/serialize.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <system_error>
#include <stdexcept>
#include <filesystem>
#include <optional>
#include <fstream>
#include <nlohmann/json.hpp>
#include "Net.hpp"

// Linux-only memmap
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

// ----------------- CLI -----------------
struct Args {
    std::string dataset;               // obrigatório: Fruits360 | PKLot
    std::string data_root = "./data/datasets";
    int64_t epochs = 10;
    int64_t batch_size = 512;
    double lr = 1e-3;
    int64_t seed = 42;
    std::string dtype_arg = "float32"; // float64|float32|float16|bfloat16
    bool use_dropout = false;
    int64_t max_ram_mb = 2048;         // limite para decidir memmap fallback
    std::string load_model;            // caminho de Module completo
    std::string load_state;            // caminho de state_dict
    bool force_cpu = false;            // se true, usa CPU mesmo se GPU existir
};

static void print_usage() {
    std::cerr
        << "Usage: ./app --dataset Fruits360|PKLot [--data_root DIR] [--epochs N] [--batch_size N]\n"
        << "             [--lr LR] [--dtype float64|float32|float16|bfloat16] [--seed S]\n"
        << "             [--use_dropout] [--max_ram_mb MB] [--load_model PATH] [--load_state PATH]\n";
}

static void print_help() {
    print_usage();
    std::cerr
        << "\n"
        << "Options:\n"
        << "  --dataset         Dataset name (Fruits360 or PKLot) [required]\n"
        << "  --data_root      Root directory for datasets (default: ./data/datasets)\n"
        << "  --epochs         Number of training epochs (default: 10)\n"
        << "  --batch_size     Mini-batch size (default: 512)\n"
        << "  --lr             Learning rate (default: 1e-3)\n"
        << "  --dtype          Data type: float64, float32, float16, bfloat16 (default: float32)\n"
        << "  --seed           Random seed (default: 42)\n"
        << "  --use_dropout    Use dropout layer before fully connected layers\n"
        << "  --max_ram_mb     Max RAM in MB before using memmap fallback (default: 2048)\n"
        << "  --load_model     Path to load full TorchScript Module\n"
        << "  --load_state     Path to load state_dict saved from Python\n"
        << "  --cpu            Force using CPU even if GPU is available\n";
}

static std::optional<Args> parse_args(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;++i) {
        std::string k = argv[i];
        auto need = [&](int i){ return i+1<argc; };
        if (k == "--dataset" && need(i)) { a.dataset = argv[++i]; }
        else if (k == "--data_root" && need(i)) { a.data_root = argv[++i]; }
        else if (k == "--epochs" && need(i)) { a.epochs = std::stoll(argv[++i]); }
        else if (k == "--batch_size" && need(i)) { a.batch_size = std::stoll(argv[++i]); }
        else if (k == "--lr" && need(i)) { a.lr = std::stod(argv[++i]); }
        else if (k == "--dtype" && need(i)) { a.dtype_arg = argv[++i]; }
        else if (k == "--seed" && need(i)) { a.seed = std::stoll(argv[++i]); }
        else if (k == "--use_dropout") { a.use_dropout = true; }
        else if (k == "--max_ram_mb" && need(i)) { a.max_ram_mb = std::stoll(argv[++i]); }
        else if (k == "--load_model" && need(i)) { a.load_model = argv[++i]; }
        else if (k == "--load_state" && need(i)) { a.load_state = argv[++i]; }
        else if (k == "--cpu") { a.force_cpu = true; }
        else {
            std::cerr << "Unknown or incomplete arg: " << k << "\n";
            print_help();
            return std::nullopt;
        }
    }
    if (a.dataset.empty()) {
        std::cerr << "[!] --dataset is required.\n";
        print_help();
        return std::nullopt;
    }
    return a;
}

// ----------------- Utils -----------------
static torch::Dtype parse_dtype(const std::string& s) {
    if (s == "float64") return torch::kFloat64;
    if (s == "float32") return torch::kFloat32;
    if (s == "float16") return torch::kFloat16;
    if (s == "bfloat16") return torch::kBFloat16;
    // default
    std::cerr << "[!] Unknown dtype '" << s << "', defaulting to float32.\n";
    return torch::kFloat32;
}

static int64_t bytes_for_tensor(int64_t n, int64_t c, int64_t h, int64_t w, int bytes_per_elem) {
    return n * c * h * w * bytes_per_elem;
}

static int64_t bytes_for_labels(int64_t n, int bytes_per_elem) {
    return n * bytes_per_elem;
}

// Helper: load a Tensor saved via Python torch.save(tensor, path)
static torch::Tensor load_tensor_pickle(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open tensor file: " + path.string());
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto ivalue = torch::pickle_load(buffer);
    if (!ivalue.isTensor()) {
        throw std::runtime_error("Loaded object is not a Tensor for file: " + path.string());
    }
    return ivalue.toTensor();
}

// ----------------- Metadata load -----------------
struct MetaInfo {
    int64_t num_classes = 0;
    int64_t image_h = 64;
    int64_t image_w = 64;
    int64_t train_size = 0;
    int64_t val_size = 0;
    int64_t test_size = 0;
    bool memmap_train = false;
    bool memmap_val = false;
    bool memmap_test = false;
    std::vector<double> norm_mean;
    std::vector<double> norm_std;
};

static MetaInfo load_metadata(const fs::path& meta_path) {
    using nlohmann::json;
    MetaInfo info;
    std::ifstream in(meta_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open metadata.json at " + meta_path.string());
    }
    json j;
    in >> j;

    if (j.contains("classes") && j["classes"].is_array()) {
        info.num_classes = static_cast<int64_t>(j["classes"].size());
    }
    if (j.contains("image_size") && j["image_size"].is_array() && j["image_size"].size() == 2) {
        info.image_h = j["image_size"][0].get<int64_t>();
        info.image_w = j["image_size"][1].get<int64_t>();
    }
    if (j.contains("train_size")) info.train_size = j["train_size"].get<int64_t>();
    if (j.contains("val_size"))   info.val_size   = j["val_size"].get<int64_t>();
    if (j.contains("test_size"))  info.test_size  = j["test_size"].get<int64_t>();

    if (j.contains("memmap_used") && j["memmap_used"].is_object()) {
        auto mm = j["memmap_used"];
        if (mm.contains("train")) info.memmap_train = mm["train"].get<bool>();
        if (mm.contains("val"))   info.memmap_val   = mm["val"].get<bool>();
        if (mm.contains("test"))  info.memmap_test  = mm["test"].get<bool>();
    }

    if (j.contains("norm_mean")) {
        for (auto& v : j["norm_mean"]) info.norm_mean.push_back(v.get<double>());
    }
    if (j.contains("norm_std")) {
        for (auto& v : j["norm_std"]) info.norm_std.push_back(v.get<double>());
    }

    return info;
}

// ----------------- Memmap RAII -----------------
struct MMap {
    void* addr = MAP_FAILED;
    size_t length = 0;
    int fd = -1;
    bool readwrite = false;

    static MMap map_file(const std::string& path, bool rw=false) {
        MMap m;
        m.readwrite = rw;
        m.fd = ::open(path.c_str(), rw ? O_RDWR : O_RDONLY);
        if (m.fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open failed for " + path);
        }
        struct stat st{};
        if (fstat(m.fd, &st) != 0) {
            int err = errno;
            ::close(m.fd);
            throw std::system_error(err, std::generic_category(), "fstat failed for " + path);
        }
        m.length = static_cast<size_t>(st.st_size);
        m.addr = ::mmap(nullptr, m.length, rw ? (PROT_READ|PROT_WRITE) : PROT_READ, MAP_SHARED, m.fd, 0);
        if (m.addr == MAP_FAILED) {
            int err = errno;
            ::close(m.fd);
            throw std::system_error(err, std::generic_category(), "mmap failed for " + path);
        }
        return m;
    }

    ~MMap() {
        if (addr != MAP_FAILED) ::munmap(addr, length);
        if (fd >= 0) ::close(fd);
    }

    template<typename T>
    const T* data_as() const {
        return reinterpret_cast<const T*>(addr);
    }

    template<typename T>
    T* data_as() {
        return reinterpret_cast<T*>(addr);
    }
};

// ----------------- Datasets -----------------
class InMemoryTensorDataset : public torch::data::datasets::Dataset<InMemoryTensorDataset> {
    torch::Tensor images_; // [N, C, H, W], uint8 or float
    torch::Tensor labels_; // [N], int64
    torch::Tensor indices_; // [M] optional subset
public:
    InMemoryTensorDataset(torch::Tensor images, torch::Tensor labels, torch::Tensor indices = {})
        : images_(std::move(images)), labels_(std::move(labels)), indices_(std::move(indices)) {
        if (!indices_.defined() || indices_.numel() == 0) {
            indices_ = torch::arange(images_.size(0), torch::dtype(torch::kInt64));
        }
    }

    torch::data::Example<> get(size_t idx) override {
        auto i = indices_[static_cast<long>(idx)].item<int64_t>();
        auto img = images_[i];   // [C,H,W]
        auto lbl = labels_[i];   // scalar int64 tensor
        return {img, lbl};
    }

    torch::optional<size_t> size() const override {
        return static_cast<size_t>(indices_.size(0));
    }
};

class MemmapTensorDataset : public torch::data::datasets::Dataset<MemmapTensorDataset> {
    // File-backed memmaps
    std::shared_ptr<MMap> data_map_;
    std::shared_ptr<MMap> label_map_;
    int64_t n_, c_, h_, w_;
    torch::Tensor indices_;
public:
    MemmapTensorDataset(std::shared_ptr<MMap> data_map,
                        std::shared_ptr<MMap> label_map,
                        int64_t n, int64_t c, int64_t h, int64_t w,
                        torch::Tensor indices = {})
        : data_map_(std::move(data_map)), label_map_(std::move(label_map)),
          n_(n), c_(c), h_(h), w_(w), indices_(std::move(indices)) {
        if (!indices_.defined() || indices_.numel() == 0) {
            indices_ = torch::arange(n_, torch::dtype(torch::kInt64));
        }
        // Quick sanity on file sizes
        size_t expect_data_bytes = static_cast<size_t>(n_) * c_ * h_ * w_ * sizeof(uint8_t);
        size_t expect_lab_bytes  = static_cast<size_t>(n_) * sizeof(int64_t);
        if (data_map_->length != expect_data_bytes) {
            std::cerr << "[!] Warning: data memmap size mismatch. File=" << data_map_->length
                      << " expected=" << expect_data_bytes << "\n";
        }
        if (label_map_->length != expect_lab_bytes) {
            std::cerr << "[!] Warning: label memmap size mismatch. File=" << label_map_->length
                      << " expected=" << expect_lab_bytes << "\n";
        }
    }

    torch::data::Example<> get(size_t idx) override {
        auto i = indices_[static_cast<long>(idx)].item<int64_t>();
        // Compute base offsets
        const uint8_t* base = data_map_->data_as<uint8_t>();
        const int64_t* lbase = label_map_->data_as<int64_t>();

        size_t img_offset = static_cast<size_t>(i) * c_ * h_ * w_;
        const uint8_t* ptr = base + img_offset;
        const int64_t* lptr = lbase + i;

        // Wrap without ownership; lifetime tied to shared_ptr maps
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto img = torch::from_blob(const_cast<uint8_t*>(ptr), {c_, h_, w_}, options).clone(); // clone to make it safe for stacking
        auto lbl = torch::from_blob(const_cast<int64_t*>(lptr), {}, torch::TensorOptions().dtype(torch::kInt64)).clone();

        return {img, lbl};
    }

    torch::optional<size_t> size() const override {
        return static_cast<size_t>(indices_.size(0));
    }
};

// ----------------- Accuracy -----------------
static double compute_accuracy(torch::Tensor outputs, torch::Tensor labels) {
    auto preds = outputs.argmax(1);
    auto correct = preds.eq(labels).to(torch::kInt64).sum().item<int64_t>();
    auto total = labels.size(0);
    return total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
}

// Interface de type-erasure para diferentes instâncias de StatelessDataLoader<...>
struct ILoader {
    virtual ~ILoader() = default;
    virtual torch::data::Iterator<typename std::vector<torch::data::Example<>>::value_type> begin() = 0;
    virtual torch::data::Iterator<typename std::vector<torch::data::Example<>>::value_type> end() = 0;
};

template <typename ConcreteLoader>
struct LoaderHolder : ILoader {
    std::unique_ptr<ConcreteLoader> loader_;
    explicit LoaderHolder(std::unique_ptr<ConcreteLoader> l) : loader_(std::move(l)) {}
    torch::data::Iterator<typename std::vector<torch::data::Example<>>::value_type> begin() override {
        return loader_->begin();
    }
    torch::data::Iterator<typename std::vector<torch::data::Example<>>::value_type> end() override {
        return loader_->end();
    }
};

struct DatasetPack {
    std::unique_ptr<ILoader> train_loader;
    std::unique_ptr<ILoader> val_loader;
    std::unique_ptr<ILoader> test_loader;
};

// Decide backend por split (memmap ou .pt) e cria o LoaderHolder apropriado
static std::unique_ptr<ILoader> make_split_loader(
    const Args& args,
    const MetaInfo& meta,
    const fs::path& processed_dir,
    const std::string& split_name,
    int64_t split_size,
    const torch::data::DataLoaderOptions& opts) {

    const int64_t C = 3;
    const int64_t H = meta.image_h;
    const int64_t W = meta.image_w;

    auto bytes_for_split = [&](int64_t n) {
        return bytes_for_tensor(n, C, H, W, 1) + bytes_for_labels(n, 8);
    };

    fs::path images_pt = processed_dir / (split_name + "_images.pt");
    fs::path labels_pt = processed_dir / (split_name + "_labels.pt");
    fs::path images_mm = processed_dir / (split_name + "_images.memmap");
    fs::path labels_mm = processed_dir / (split_name + "_labels.memmap");

    bool have_memmap = fs::exists(images_mm) && fs::exists(labels_mm) && split_size > 0;

    bool use_memmap = false;
    if (have_memmap) {
        auto bytes = bytes_for_split(split_size);
        auto mb = bytes / (1024 * 1024);
        use_memmap = (mb > args.max_ram_mb);
    }

    if (use_memmap) {
        auto dm = std::make_shared<MMap>(MMap::map_file(images_mm.string(), false));
        auto lm = std::make_shared<MMap>(MMap::map_file(labels_mm.string(), false));
        auto ds = MemmapTensorDataset(dm, lm, split_size, C, H, W);
        auto map = ds.map(torch::data::transforms::Stack<>());
        auto loader = torch::data::make_data_loader(std::move(map), opts);
        using LoaderT = std::remove_reference_t<decltype(*loader)>;
        return std::make_unique<LoaderHolder<LoaderT>>(std::move(loader));
    }

    torch::Tensor X, Y;
    X = load_tensor_pickle(images_pt);
    Y = load_tensor_pickle(labels_pt);
    if (X.dtype() != torch::kUInt8) X = X.to(torch::kUInt8);
    if (Y.dtype() != torch::kInt64) Y = Y.to(torch::kInt64);

    auto ds = InMemoryTensorDataset(X, Y);
    auto map = ds.map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader(std::move(map), opts);
    using LoaderT = std::remove_reference_t<decltype(*loader)>;
    return std::make_unique<LoaderHolder<LoaderT>>(std::move(loader));
}

// build_loaders mantém lógica, mas usa make_data_loader corretamente e encapsula em LoaderHolder
static DatasetPack build_loaders(const Args& args, const MetaInfo& meta,
                                 const fs::path& processed_dir) {
    const int64_t C = 3, H = meta.image_h, W = meta.image_w;
    auto bytes_for_split = [&](int64_t n) {
        return bytes_for_tensor(n, C, H, W, 1) + bytes_for_labels(n, 8);
    };

    int64_t train_bytes = bytes_for_split(meta.train_size);
    int64_t val_bytes   = bytes_for_split(meta.val_size);
    int64_t test_bytes  = bytes_for_split(meta.test_size);
    int64_t total_mb = (train_bytes + val_bytes + test_bytes) / (1024 * 1024);

    std::cout << "[i] Estimated total dataset size ~" << total_mb
              << "MB (limit max_ram_mb=" << args.max_ram_mb << ")\n";

    torch::data::DataLoaderOptions opts;
    opts.batch_size(args.batch_size).workers(0);

    DatasetPack pack;

    struct SplitInfo {
        const char* name;
        int64_t size;
        std::unique_ptr<ILoader>* target;
    };

    SplitInfo splits[] = {
        {"train", meta.train_size, &pack.train_loader},
        {"val",   meta.val_size,   &pack.val_loader},
        {"test",  meta.test_size,  &pack.test_loader},
    };

    for (const auto& s : splits) {
        *(s.target) = make_split_loader(args, meta, processed_dir, s.name, s.size, opts);
    }

    return pack;
}

// Avaliação usando interface ILoader
static double eval_epoch(Net& model, ILoader& loader_iface,
                         torch::nn::CrossEntropyLoss& criterion,
                         torch::Device device, torch::Dtype model_dtype,
                         const MetaInfo& meta) {
    torch::NoGradGuard ng;
    model->eval();
    double loss_sum = 0.0;
    int64_t total = 0;
    for (auto it = loader_iface.begin(); it != loader_iface.end(); ++it) {
        auto batch = *it; // Example<>
        auto inputs = batch.data.to(device);
        auto labels = batch.target.to(device).to(torch::kInt64);
        // Se vier como uint8 (caso memmap), converte para float32 [0,1]
        if (inputs.dtype() == torch::kUInt8) {
            inputs = inputs.to(torch::kFloat32) / 255.0f;
            if (meta.norm_mean.size() == 3 && meta.norm_std.size() == 3) {
                auto mean = torch::tensor({static_cast<float>(meta.norm_mean[0]),
                                           static_cast<float>(meta.norm_mean[1]),
                                           static_cast<float>(meta.norm_mean[2])})
                                .view({1,3,1,1})
                                .to(inputs.device(), inputs.dtype());
                auto std = torch::tensor({static_cast<float>(meta.norm_std[0]),
                                          static_cast<float>(meta.norm_std[1]),
                                          static_cast<float>(meta.norm_std[2])})
                               .view({1,3,1,1})
                               .to(inputs.device(), inputs.dtype());
                inputs = (inputs - mean) / std;
            }
        }
        if (inputs.dtype() != model_dtype) inputs = inputs.to(model_dtype);
        auto outputs = model->forward(inputs);
        auto loss = criterion(outputs, labels);
        loss_sum += loss.item<double>() * inputs.size(0);
        total += inputs.size(0);
    }
    return (total == 0) ? 0.0 : loss_sum / static_cast<double>(total);
}

static double eval_accuracy(Net& model, ILoader& loader_iface,
                            torch::Device device, torch::Dtype model_dtype,
                            const MetaInfo& meta) {
    torch::NoGradGuard ng;
    model->eval();
    int64_t total = 0, correct = 0;
    for (auto it = loader_iface.begin(); it != loader_iface.end(); ++it) {
        auto batch = *it;
        auto inputs = batch.data.to(device);
        auto labels = batch.target.to(device).to(torch::kInt64);
        // Se vier como uint8 (caso memmap), converte para float32 [0,1]
        if (inputs.dtype() == torch::kUInt8) {
            inputs = inputs.to(torch::kFloat32) / 255.0f;
            if (meta.norm_mean.size() == 3 && meta.norm_std.size() == 3) {
                auto mean = torch::tensor({static_cast<float>(meta.norm_mean[0]),
                                           static_cast<float>(meta.norm_mean[1]),
                                           static_cast<float>(meta.norm_mean[2])})
                                .view({1,3,1,1})
                                .to(inputs.device(), inputs.dtype());
                auto std = torch::tensor({static_cast<float>(meta.norm_std[0]),
                                          static_cast<float>(meta.norm_std[1]),
                                          static_cast<float>(meta.norm_std[2])})
                               .view({1,3,1,1})
                               .to(inputs.device(), inputs.dtype());
                inputs = (inputs - mean) / std;
            }
        }
        if (inputs.dtype() != model_dtype) inputs = inputs.to(model_dtype);
        auto outputs = model->forward(inputs);
        auto preds = outputs.argmax(1);
        correct += preds.eq(labels).to(torch::kInt64).sum().item<int64_t>();
        total += inputs.size(0);
    }
    return (total == 0) ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
}

static void save_state_dict_json(const Net& model, const std::string& path) {
    using nlohmann::json;
    json root;

    for (const auto& p : model->named_parameters()) {
        const auto& name = p.key();
        const auto& tensor = p.value().cpu();

        json entry;
        entry["shape"] = tensor.sizes().vec();

        std::vector<float> data(tensor.numel());
        std::memcpy(data.data(), tensor.data_ptr(), tensor.numel() * sizeof(float));
        entry["data"] = data;

        root[name] = entry;
    }

    std::ofstream f(path);
    f << root.dump(2);
}

// ----------------- Main -----------------
int main(int argc, char** argv) {
    auto pargs = parse_args(argc, argv);
    if (!pargs) return 1;
    Args args = *pargs;

    // Device: GPU por padrão, CPU se --cpu ou sem CUDA
    torch::Device device(torch::kCPU);
    if (!args.force_cpu && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
        std::cout << "[i] Using CUDA device 0" << "\n";
    } else {
        std::cout << "[i] Using CPU" << "\n";
    }

    // Seed
    torch::manual_seed(args.seed);
    std::srand(static_cast<unsigned>(args.seed));
    torch::set_num_threads(1);

    // Dtype
    auto dtype = parse_dtype(args.dtype_arg);
    if ((dtype == torch::kFloat16 || dtype == torch::kBFloat16) && device.is_cpu()) {
        std::cerr << "[!] Warning: 16-bit dtypes on CPU may be slow or unsupported for some ops.\n";
    }

    // Resolve processed dir
    fs::path processed_dir = fs::path(args.data_root) / "processed" / args.dataset;
    fs::path meta_path = processed_dir / "metadata.json";
    if (!fs::exists(meta_path)) {
        std::cerr << "[!] metadata.json not found at " << meta_path << "\n";
        return 2;
    }
    auto meta = load_metadata(meta_path);
    std::cout << "[i] Dataset=" << args.dataset
              << " classes=" << meta.num_classes
              << " image_size=" << meta.image_h << "x" << meta.image_w
              << " train=" << meta.train_size << " test=" << meta.test_size
              << " memmap_present=" << std::boolalpha << meta.memmap_train << "\n";

    auto loaders = build_loaders(args, meta, processed_dir);

    // Build model
    std::shared_ptr<Net> model = std::make_shared<Net>(dtype, device, /*in_channels=*/3, /*num_classes=*/meta.num_classes, args.use_dropout, /*base_filters=*/32);

    // Load pretrained if provided
    if (!args.load_model.empty()) {
        try {
            torch::load(*model, args.load_model);
            std::cout << "[i] Loaded full Module from: " << args.load_model << "\n";
            // Ensure dtype/device set as requested after load
            model->get()->to_dtype(dtype, device);
        } catch (const std::exception& e) {
            std::cerr << "[!] Failed to load full model: " << e.what() << "\n";
            return 3;
        }
    } else if (!args.load_state.empty()) {
        try {
            torch::serialize::InputArchive archive;
            archive.load_from(args.load_state);
            model->get()->load(archive);
            std::cout << "[i] Loaded state_dict from: " << args.load_state << "\n";
            model->get()->to_dtype(dtype, device);
        } catch (const std::exception& e) {
            std::cerr << "[!] Failed to load state_dict: " << e.what() << "\n";
            return 3;
        }
    }

    // Optimizer and loss
    torch::optim::Adam optimizer(model->get()->parameters(), torch::optim::AdamOptions(args.lr));
    auto criterion = torch::nn::CrossEntropyLoss();

    std::cout << "Training started. dtype=" << args.dtype_arg
              << " epochs=" << args.epochs
              << " batch_size=" << args.batch_size
              << "\n";

    auto t_start = std::chrono::steady_clock::now();

    for (int64_t epoch = 1; epoch <= args.epochs; ++epoch) {
        model->get()->train();
        double epoch_loss_sum = 0.0;
        int64_t seen = 0;

        auto& train_loader = *loaders.train_loader;
        for (auto& batch : train_loader) {
            auto inputs = batch.data.to(device);
            auto labels = batch.target.to(device).to(torch::kInt64);

            // Se vier como uint8 (caso memmap), converte para float32 [0,1]
            if (inputs.dtype() == torch::kUInt8) {
                inputs = inputs.to(torch::kFloat32) / 255.0f;
                if (meta.norm_mean.size() == 3 && meta.norm_std.size() == 3) {
                    auto mean = torch::tensor({static_cast<float>(meta.norm_mean[0]),
                                               static_cast<float>(meta.norm_mean[1]),
                                               static_cast<float>(meta.norm_mean[2])})
                                    .view({1,3,1,1})
                                    .to(inputs.device(), inputs.dtype());
                    auto std = torch::tensor({static_cast<float>(meta.norm_std[0]),
                                              static_cast<float>(meta.norm_std[1]),
                                              static_cast<float>(meta.norm_std[2])})
                                   .view({1,3,1,1})
                                   .to(inputs.device(), inputs.dtype());
                    inputs = (inputs - mean) / std;
                }
            }
            if (inputs.dtype() != dtype) inputs = inputs.to(dtype);

            optimizer.zero_grad();
            auto outputs = model->get()->forward(inputs);
            auto loss = criterion(outputs, labels);
            loss.backward();
            optimizer.step();

            epoch_loss_sum += loss.item<double>() * inputs.size(0);
            seen += inputs.size(0);
        }

        double train_loss = (seen > 0) ? (epoch_loss_sum / static_cast<double>(seen)) : 0.0;
        double val_loss = eval_epoch(*model, *loaders.val_loader, criterion, device, dtype, meta);
        double val_acc  = eval_accuracy(*model, *loaders.val_loader, device, dtype, meta);

        std::cout << "Epoch " << epoch
                  << " | train_loss=" << train_loss
                  << " | val_loss=" << val_loss
                  << " | val_acc=" << val_acc
                  << "\n";
    }

    auto t_end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "Training finished. Total time ms=" << total_ms << "\n";

    // Final test
    double test_loss = eval_epoch(*model, *loaders.test_loader, criterion, device, dtype, meta);
    double test_acc  = eval_accuracy(*model, *loaders.test_loader, device, dtype, meta);
    std::cout << "Test loss=" << test_loss << " test_acc=" << test_acc << "\n";

    // Define model and JSON save paths
    fs::path model_dir = fs::path(args.data_root) / ".." / "models" / args.dataset / "test";
    fs::create_directories(model_dir);
    fs::path model_path = model_dir / "model_full.pt";
    fs::path state_dict_path = model_dir / "state_dict.json";
    fs::path metrics_path = model_dir / "metrics.json";

    // Save evaluation metrics to JSON alongside the model
    try {
        nlohmann::json jres;
        jres["dataset"] = args.dataset;
        jres["dtype"] = args.dtype_arg;
        jres["epochs"] = args.epochs;
        jres["batch_size"] = args.batch_size;
        jres["lr"] = args.lr;
        jres["test_loss"] = test_loss;
        jres["test_acc"] = test_acc;
        std::ofstream jm(metrics_path);
        jm << jres.dump(4) << std::endl;
        std::cout << "[i] Saved metrics to " << metrics_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[!] Failed to save metrics: " << e.what() << "\n";
    }

    // Save model
    try {
        torch::save(*model, model_path.string());
        std::cout << "[i] Saved " << model_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[!] Failed to save model: " << e.what() << "\n";
    }

    try {
        save_state_dict_json(*model, state_dict_path.string());
        std::cout << "[i] Saved model state_dict to " << state_dict_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[!] Failed to save state_dict: " << e.what() << "\n";
    }

    return 0;
}
