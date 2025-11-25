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

    // Clonar para RAM de forma contígua
    if (float32_) {
        images_out = torch::from_blob(
            img_ptr,
            { static_cast<long>(chunk_size), static_cast<long>(C_), static_cast<long>(H_), static_cast<long>(W_) },
            torch::kFloat32
        ).clone();
    } else {
        images_out = torch::from_blob(
            img_ptr,
            { static_cast<long>(chunk_size), static_cast<long>(C_), static_cast<long>(H_), static_cast<long>(W_) },
            torch::kUInt8
        ).clone();
    }

    labels_out = torch::from_blob(
        lbl_ptr,
        { static_cast<long>(chunk_size) },
        torch::kInt64  // labels são uint64 no arquivo, mas LibTorch usa int64
    ).clone();

    // libera mmaps temporários
    munmap(img_ptr, img_bytes);
    munmap(lbl_ptr, lbl_bytes);

}
