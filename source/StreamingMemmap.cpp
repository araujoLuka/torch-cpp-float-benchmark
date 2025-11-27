#include "StreamingMemmap.hpp"

static inline size_t get_file_size(int fd) {
    struct stat st;
    if (fstat(fd, &st) != 0)
        throw std::runtime_error(std::string("fstat failed: ") + strerror(errno));
    return (size_t)st.st_size;
}

StreamingMemmap::StreamingMemmap(const std::string& images_path,
                                 const std::string& labels_path,
                                 size_t N,
                                 size_t C,
                                 size_t H,
                                 size_t W,
                                 bool images_are_float32)
    : N_(N), C_(C), H_(H), W_(W), float32_(images_are_float32)
{
    element_size_ = float32_ ? 4 : 1;
    sample_bytes_ = C_ * H_ * W_ * element_size_;

    img_fd_ = open(images_path.c_str(), O_RDONLY);
    if (img_fd_ < 0)
        throw std::runtime_error("Failed to open images file: " + std::string(strerror(errno)));

    lbl_fd_ = open(labels_path.c_str(), O_RDONLY);
    if (lbl_fd_ < 0) {
        close(img_fd_);
        throw std::runtime_error("Failed to open labels file: " + std::string(strerror(errno)));
    }

    // Tensores estáticos para reutilizar memória entre chunks.
    // Começam vazios e são redimensionados conforme necessário em load_chunk.
    if (float32_) {
        images_buffer_ = torch::empty({0, static_cast<long>(C_), static_cast<long>(H_), static_cast<long>(W_)}, torch::kFloat32);
    } else {
        images_buffer_ = torch::empty({0, static_cast<long>(C_), static_cast<long>(H_), static_cast<long>(W_)}, torch::kUInt8);
    }
    labels_buffer_ = torch::empty({0}, torch::kInt64);
}

StreamingMemmap::~StreamingMemmap() {
    if (img_fd_ >= 0) close(img_fd_);
    if (lbl_fd_ >= 0) close(lbl_fd_);
}

void StreamingMemmap::load_chunk(size_t offset, size_t chunk_size, torch::Tensor& images_out, torch::Tensor& labels_out) {
    if (offset >= N_)
        throw std::out_of_range("Offset outside dataset");

    if (offset + chunk_size > N_)
        chunk_size = N_ - offset;

    size_t img_off  = offset * sample_bytes_;
    size_t img_bytes = chunk_size * sample_bytes_;

    size_t lbl_off  = offset * sizeof(uint64_t);
    size_t lbl_bytes = chunk_size * sizeof(uint64_t);

    // mmap da fatia
    void* img_ptr = mmap(nullptr, img_bytes, PROT_READ, MAP_SHARED, img_fd_, img_off);
    if (img_ptr == MAP_FAILED)
        throw std::runtime_error("mmap images chunk failed: " + std::string(strerror(errno)));

    void* lbl_ptr = mmap(nullptr, lbl_bytes, PROT_READ, MAP_SHARED, lbl_fd_, lbl_off);
    if (lbl_ptr == MAP_FAILED) {
        munmap(img_ptr, img_bytes);
        throw std::runtime_error("mmap labels chunk failed: " + std::string(strerror(errno)));
    }

    // Garante que os buffers estáticos têm a capacidade correta
    const auto needed_img_sizes = std::vector<int64_t>{
        static_cast<int64_t>(chunk_size),
        static_cast<int64_t>(C_),
        static_cast<int64_t>(H_),
        static_cast<int64_t>(W_)
    };

    if (!images_buffer_.defined() || images_buffer_.sizes() != torch::IntArrayRef(needed_img_sizes)) {
        if (float32_) {
            images_buffer_ = torch::empty(needed_img_sizes, torch::kFloat32);
        } else {
            images_buffer_ = torch::empty(needed_img_sizes, torch::kUInt8);
        }
    }

    if (!labels_buffer_.defined() || labels_buffer_.size(0) != static_cast<long>(chunk_size)) {
        labels_buffer_ = torch::empty({ static_cast<long>(chunk_size) }, torch::kInt64);
    }

    // Copia diretamente dos mmaps para os buffers reutilizáveis
    void* img_dest = images_buffer_.data_ptr();
    std::memcpy(img_dest, img_ptr, img_bytes);

    void* lbl_dest = labels_buffer_.data_ptr();
    std::memcpy(lbl_dest, lbl_ptr, lbl_bytes);

    // libera mmaps temporários
    munmap(img_ptr, img_bytes);
    munmap(lbl_ptr, lbl_bytes);

    // Retorna views dos buffers estáticos
    images_out = images_buffer_;
    labels_out = labels_buffer_;
}
