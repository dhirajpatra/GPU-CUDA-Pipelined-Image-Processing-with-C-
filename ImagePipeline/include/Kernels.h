// Kernels.h - host-side interface to the GPU kernels.
//
// Pixel formats used on the device:
//   * colour  : interleaved BGR, 3 bytes/pixel (same order as BMP, so no swizzle on upload)
//   * gray    : 1 byte/pixel
//   * edge    : 1 byte/pixel, |Laplacian response| clamped to [0,255]
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "CudaUtils.h"

namespace ipp {

/// One detected object (an 8-connected set of edge pixels).
struct Detection {
    float cx;            ///< Centroid x (column), pixel-centre coordinates.
    float cy;            ///< Centroid y (row).
    std::uint32_t area;  ///< Number of edge pixels in the component.
};

/// Result of the final stage for one image.
struct AnalysisResult {
    std::vector<Detection> objects;  ///< Sorted by (cy, cx) for deterministic output. May be truncated.
    std::uint32_t totalObjects = 0;  ///< True number of objects (>= objects.size() if truncated).
    bool truncated = false;
    int labelingRounds = 0;          ///< Convergence checks needed by connected-component labeling.
};

/// Normalised 3x3 Gaussian kernel, row-major. sigma must be > 0.
inline std::array<float, 9> makeGaussianKernel3x3(float sigma) {
    std::array<float, 9> k{};
    float sum = 0.0f;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const float v = std::exp(-static_cast<float>(x * x + y * y) / (2.0f * sigma * sigma));
            k[static_cast<std::size_t>((y + 1) * 3 + (x + 1))] = v;
            sum += v;
        }
    }
    for (float& v : k) v /= sum;
    return k;
}

/// Uploads the Gaussian kernel to constant memory. Call once, before any stream uses launchGaussian.
void setGaussianSigma(float sigma);

/// BGR -> gray conversion fused with a 3x3 Gaussian blur. Border pixels use clamp-to-edge.
void launchGaussian(const std::uint8_t* dBgr, std::uint8_t* dGray, int width, int height, cudaStream_t stream);

/// 4-neighbour Laplacian [0 1 0; 1 -4 1; 0 1 0]; stores |response| clamped to 255.
void launchLaplacian(const std::uint8_t* dGray, std::uint8_t* dEdge, int width, int height, cudaStream_t stream);

/// Threshold + connected-component labeling + per-object statistics + colour overlay.
///
/// Owns all scratch memory, so it must be used by one thread at a time (the final
/// pipeline stage is single-threaded by design). analyze() synchronises `stream`
/// before returning the host-side results.
class ComponentAnalyzer {
public:
    explicit ComponentAnalyzer(std::size_t maxPixels, int maxObjects = 4096);

    /// Edge pixels are those with dEdge >= threshold. Components smaller than `minArea`
    /// pixels are treated as noise (not reported, not drawn). Edges are painted in place
    /// on dBgr and a cross is drawn on each centroid.
    AnalysisResult analyze(const std::uint8_t* dEdge, std::uint8_t* dBgr, int width, int height,
                           std::uint8_t threshold, std::uint32_t minArea, cudaStream_t stream);

private:
    std::size_t capacity_;
    int maxObjects_;
    DeviceBuffer<int> label_;
    DeviceBuffer<std::uint32_t> count_;
    DeviceBuffer<unsigned long long> sumX_;
    DeviceBuffer<unsigned long long> sumY_;
    DeviceBuffer<Detection> detections_;
    DeviceBuffer<int> flags_;  // [0] = "labels changed", [1] = object counter
    PinnedBuffer<int> hostFlags_;
    PinnedBuffer<Detection> hostDetections_;
};

}  // namespace ipp
