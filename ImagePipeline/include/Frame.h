// Frame.h - one "slot" of the frame pool. A Frame owns every buffer an image
// needs while travelling through the pipeline, so no memory is allocated in
// steady state and the number of Frames bounds the number of images in flight.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "CudaUtils.h"
#include "Kernels.h"
#include "Stats.h"

namespace ipp {

struct StageEvents {
    CudaEvent start;
    CudaEvent stop;
};

struct Frame {
    // Identity / geometry of the image currently held by this slot.
    std::size_t index = 0;
    std::string name;
    int width = 0;
    int height = 0;

    // Memory. Pinned host buffers enable async DMA; device buffers are reused every frame.
    PinnedBuffer<std::uint8_t> hostIn;
    PinnedBuffer<std::uint8_t> hostOut;
    DeviceBuffer<std::uint8_t> dBgr;
    DeviceBuffer<std::uint8_t> dGray;
    DeviceBuffer<std::uint8_t> dEdge;

    // GPU timing events, one start/stop pair per stage. stop[s] doubles as the
    // dependency that stage s+1 waits on (cudaStreamWaitEvent), so stages never block the CPU.
    StageEvents events[kNumStages];

    AnalysisResult result;

    // Host-side timing.
    std::chrono::steady_clock::time_point startedAt;
    double readMs = 0.0;

    std::size_t pixels() const { return static_cast<std::size_t>(width) * static_cast<std::size_t>(height); }

    void allocate(std::size_t maxPixels) {
        hostIn.allocate(maxPixels * 3);
        hostOut.allocate(maxPixels * 3);
        dBgr.allocate(maxPixels * 3);
        dGray.allocate(maxPixels);
        dEdge.allocate(maxPixels);
    }
};

}  // namespace ipp
