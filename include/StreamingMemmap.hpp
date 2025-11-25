#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstring>
#include <string>

class StreamingMemmap
{
    public:
     StreamingMemmap(const std::string& images_path,
                          const std::string& labels_path,
                          size_t N,
                          size_t C,
                          size_t H,
                          size_t W,
                          bool images_are_float32 = false);

     ~StreamingMemmap();

     // Mapeia uma fatia [offset, offset + chunk_size) e devolve tensores já clonados para RAM.
     void load_chunk(size_t offset, size_t chunk_size, torch::Tensor& images_out, torch::Tensor& labels_out);

     size_t total_size() const { return N_; }

    private:
    int img_fd_ = -1;
    int lbl_fd_ = -1;

    size_t N_, C_, H_, W_;
    size_t sample_bytes_;
    size_t element_size_;

    bool float32_;
        // Buffers estáticos reutilizáveis para reduzir alocações por chunk
        torch::Tensor images_buffer_;
        torch::Tensor labels_buffer_;
};
