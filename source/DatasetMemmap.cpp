#include "DatasetMemmap.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <cerrno>
#include <cstring>

static inline size_t file_size(int fd) {
    struct stat st;
    if (fstat(fd, &st) != 0) throw std::runtime_error(std::string("fstat failed: ") + std::strerror(errno));
    return static_cast<size_t>(st.st_size);
}

MemmapTensorDataset::MemmapTensorDataset(const std::string& images_path,
                                       const std::string& labels_path,
                                       size_t N,
                                       size_t C,
                                       size_t H,
                                       size_t W,
                                       ImageDtype dtype,
                                       bool force_clone_in_get)
    : images_path_(images_path), labels_path_(labels_path), N_(N), C_(C), H_(H), W_(W), dtype_(dtype), force_clone_in_get_(force_clone_in_get) {

    element_size_bytes_ = (dtype_ == ImageDtype::FLOAT32) ? 4u : 1u;
    sample_bytes_ = C_ * H_ * W_ * element_size_bytes_;

    open_and_map_files();
}

MemmapTensorDataset::MemmapTensorDataset(const MemmapTensorDataset& o)
    : images_path_(o.images_path_), labels_path_(o.labels_path_),
      N_(o.N_), C_(o.C_), H_(o.H_), W_(o.W_), dtype_(o.dtype_),
      element_size_bytes_(o.element_size_bytes_), sample_bytes_(o.sample_bytes_),
      force_clone_in_get_(o.force_clone_in_get_) {
    open_and_map_files();
}

MemmapTensorDataset::MemmapTensorDataset(MemmapTensorDataset&& o) noexcept {
    *this = std::move(o);
}

MemmapTensorDataset& MemmapTensorDataset::operator=(MemmapTensorDataset&& o) noexcept {
    if (this != &o) {
        images_path_ = std::move(o.images_path_);
        labels_path_ = std::move(o.labels_path_);
        N_ = o.N_; C_ = o.C_; H_ = o.H_; W_ = o.W_; dtype_ = o.dtype_;
        element_size_bytes_ = o.element_size_bytes_;
        sample_bytes_ = o.sample_bytes_;

        images_fd_ = o.images_fd_; labels_fd_ = o.labels_fd_;
        images_map_ = o.images_map_; labels_map_ = o.labels_map_;
        images_map_bytes_ = o.images_map_bytes_; labels_map_bytes_ = o.labels_map_bytes_;
        force_clone_in_get_ = o.force_clone_in_get_;

        o.images_fd_ = -1; o.labels_fd_ = -1;
        o.images_map_ = MAP_FAILED; o.labels_map_ = MAP_FAILED;
        o.images_map_bytes_ = 0; o.labels_map_bytes_ = 0;
    }
    return *this;
}

MemmapTensorDataset::~MemmapTensorDataset() {
    if (images_map_ != MAP_FAILED && images_map_ != nullptr) {
        munmap(images_map_, images_map_bytes_);
        images_map_ = MAP_FAILED;
    }
    if (labels_map_ != MAP_FAILED && labels_map_ != nullptr) {
        munmap(labels_map_, labels_map_bytes_);
        labels_map_ = MAP_FAILED;
    }
    if (images_fd_ >= 0) { close(images_fd_); images_fd_ = -1; }
    if (labels_fd_ >= 0) { close(labels_fd_); labels_fd_ = -1; }
}

void MemmapTensorDataset::open_and_map_files() {
    images_fd_ = open(images_path_.c_str(), O_RDONLY);
    if (images_fd_ < 0) throw std::runtime_error(std::string("Failed to open images file: ") + std::strerror(errno));
    images_map_bytes_ = file_size(images_fd_);
    size_t expected_images_bytes = N_ * sample_bytes_;
    if (images_map_bytes_ < expected_images_bytes) {
        close(images_fd_); images_fd_ = -1;
        throw std::runtime_error("Images file too small for provided dimensions (expected at least " + std::to_string(expected_images_bytes) + " bytes, got " + std::to_string(images_map_bytes_) + ")");
    }
    images_map_ = mmap(nullptr, images_map_bytes_, PROT_READ, MAP_SHARED, images_fd_, 0);
    if (images_map_ == MAP_FAILED) { close(images_fd_); images_fd_ = -1; throw std::runtime_error(std::string("mmap images failed: ") + std::strerror(errno)); }

    labels_fd_ = open(labels_path_.c_str(), O_RDONLY);
    if (labels_fd_ < 0) { munmap(images_map_, images_map_bytes_); images_map_ = MAP_FAILED; close(images_fd_); images_fd_ = -1; throw std::runtime_error(std::string("Failed to open labels file: ") + std::strerror(errno)); }
    labels_map_bytes_ = file_size(labels_fd_);
    size_t expected_labels_bytes = N_ * sizeof(uint64_t);
    if (labels_map_bytes_ < expected_labels_bytes) {
        munmap(images_map_, images_map_bytes_); images_map_ = MAP_FAILED; close(images_fd_); close(labels_fd_); images_fd_ = labels_fd_ = -1;
        throw std::runtime_error("Labels file too small for provided N (expected at least " + std::to_string(expected_labels_bytes) + " bytes, got " + std::to_string(labels_map_bytes_) + ")");
    }
    labels_map_ = mmap(nullptr, labels_map_bytes_, PROT_READ, MAP_SHARED, labels_fd_, 0);
    if (labels_map_ == MAP_FAILED) { munmap(images_map_, images_map_bytes_); images_map_ = MAP_FAILED; close(images_fd_); close(labels_fd_); images_fd_ = labels_fd_ = -1; throw std::runtime_error(std::string("mmap labels failed: ") + std::strerror(errno)); }
}

torch::data::Example<> MemmapTensorDataset::get(size_t index) {
    if (index >= N_) throw std::out_of_range("Index out of range in MemmapTensorDataset::get");

    // pointer to sample start
    uint8_t* base = reinterpret_cast<uint8_t*>(images_map_);
    uint8_t* sample_ptr = base + index * sample_bytes_;

    // build tensor with from_blob -> shape {C, H, W}
    // strides for {C,H,W}: {H*W, W, 1}
    int64_t h = static_cast<int64_t>(H_);
    int64_t w = static_cast<int64_t>(W_);
    int64_t c = static_cast<int64_t>(C_);

    torch::Tensor image_tensor;
    if (dtype_ == ImageDtype::FLOAT32) {
        float* ptr = reinterpret_cast<float*>(sample_ptr);
        // strides in number of elements (not bytes)
        std::vector<int64_t> strides = {h * w, w, 1};
        image_tensor = torch::from_blob(ptr, {c, h, w}, strides, torch::TensorOptions().dtype(torch::kFloat32));
    } else {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(sample_ptr);
        std::vector<int64_t> strides = {h * w, w, 1};
        image_tensor = torch::from_blob(ptr, {c, h, w}, strides, torch::TensorOptions().dtype(torch::kUInt8));
    }

    // label
    uint8_t* labels_base = reinterpret_cast<uint8_t*>(labels_map_);
    uint64_t* label_ptr = reinterpret_cast<uint64_t*>(labels_base + index * sizeof(uint64_t));
    torch::Tensor label_tensor = torch::from_blob(label_ptr, {}, torch::TensorOptions().dtype(torch::kUInt64));

    // ensure tensors don't require grad
    image_tensor = image_tensor.set_requires_grad(false).pin_memory();
    label_tensor = label_tensor.set_requires_grad(false).pin_memory();

    // If user requested clones (safer for certain DataLoader worker setups), clone here.
    if (force_clone_in_get_) {
        // contiguous + clone to own memory (expensive)
        auto img = image_tensor.contiguous().clone();
        auto lbl = label_tensor.clone();
        img.set_requires_grad(false);
        lbl.set_requires_grad(false);
        return {img, lbl};
    }

    // Return zero-copy tensors pointing to the mmap'd memory.
    // IMPORTANT: keep mmap alive for lifetime of dataset (this object holds it).
    return {image_tensor, label_tensor};
}

torch::optional<size_t> MemmapTensorDataset::size() const {
    return static_cast<size_t>(N_);
}
