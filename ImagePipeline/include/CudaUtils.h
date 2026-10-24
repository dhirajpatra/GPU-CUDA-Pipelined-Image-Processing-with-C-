// CudaUtils.h - RAII wrappers and error handling for the CUDA runtime API.
//
// Every CUDA resource used by the pipeline (device memory, pinned host memory,
// streams, events) is owned by one of these classes, so no code path can leak
// a resource when an exception unwinds the stack.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ipp {

/// Thrown whenever a CUDA runtime call returns something other than cudaSuccess.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const char* expression, const char* file, int line)
        : std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " +
                             expression + " failed: " + cudaGetErrorString(code)) {}
};

}  // namespace ipp

/// Evaluates a CUDA call and throws ipp::CudaError on failure.
#define IPP_CUDA_CHECK(expression)                                                    \
    do {                                                                              \
        const cudaError_t ipp_status_ = (expression);                                 \
        if (ipp_status_ != cudaSuccess) {                                             \
            throw ::ipp::CudaError(ipp_status_, #expression, __FILE__, __LINE__);     \
        }                                                                             \
    } while (false)

namespace ipp {

/// Owning, move-only buffer in device (GPU) memory. allocate() only ever grows.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) { release(); swap(other); }
        return *this;
    }

    /// Ensures capacity for at least `count` elements (contents are not preserved).
    void allocate(std::size_t count) {
        if (count <= capacity_) return;
        release();
        IPP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        capacity_ = count;
    }
    T* get() const noexcept { return ptr_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    void release() noexcept {
        if (ptr_ != nullptr) cudaFree(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
    }
    void swap(DeviceBuffer& o) noexcept { std::swap(ptr_, o.ptr_); std::swap(capacity_, o.capacity_); }

    T* ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

/// Owning, move-only page-locked host buffer. Pinned memory is required for
/// truly asynchronous host<->device copies that overlap with kernel execution.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept { swap(other); }
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) { release(); swap(other); }
        return *this;
    }

    void allocate(std::size_t count) {
        if (count <= capacity_) return;
        release();
        IPP_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        capacity_ = count;
    }
    T* get() const noexcept { return ptr_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    void release() noexcept {
        if (ptr_ != nullptr) cudaFreeHost(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
    }
    void swap(PinnedBuffer& o) noexcept { std::swap(ptr_, o.ptr_); std::swap(capacity_, o.capacity_); }

    T* ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

/// Non-blocking CUDA stream (does not implicitly synchronise with the legacy default stream).
class CudaStream {
public:
    CudaStream() { IPP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~CudaStream() { cudaStreamDestroy(stream_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { IPP_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

/// CUDA event with timing enabled.
class CudaEvent {
public:
    CudaEvent() { IPP_CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept { return event_; }
    void record(cudaStream_t stream) const { IPP_CUDA_CHECK(cudaEventRecord(event_, stream)); }

    /// Milliseconds between two already-completed events recorded on the same stream.
    static float elapsedMs(const CudaEvent& start, const CudaEvent& stop) {
        float ms = 0.0f;
        IPP_CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

}  // namespace ipp
