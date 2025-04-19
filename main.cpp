#include <torch/torch.h>
#include <iostream>

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

int main() {
    torch::manual_seed(0);
    torch::Tensor tensor = torch::rand({3, 3});

    std::cout << "Original Tensor:\n" << tensor << std::endl;

    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    {
    LIKWID_MARKER_START("to_float16");
    tensor = tensor.to(torch::kFloat16);
    LIKWID_MARKER_STOP("to_float16");
    }

    LIKWID_MARKER_CLOSE;

    std::cout << "Tensor in half precision:\n" << tensor << std::endl;

    return 0;
}
