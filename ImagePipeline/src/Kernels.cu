// Kernels.cu - CUDA kernels and their host launchers.
//
// Stage 1: fused BGR->gray + 3x3 Gaussian        (shared-memory tile with 1-pixel halo)
// Stage 2: 4-neighbour Laplacian                 (shared-memory tile with 1-pixel halo)
// Stage 3: threshold -> connected components (label-equivalence with union-find
//          hooking + pointer jumping) -> per-object centroid -> colour overlay
#include <algorithm>
#include <stdexcept>

#include "Kernels.h"

namespace ipp {
namespace detail {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kHooksPerCheck = 2;     // hook+compress iterations between host convergence checks
constexpr int kMaxLabelingRounds = 4096;  // safety net; real images converge in a handful
constexpr int kCrossRadius = 4;
constexpr int kNoLabel = -1;

__constant__ float c_gauss[9];

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/// BT.601 luma from BGR, rounded to nearest.
__device__ __forceinline__ std::uint8_t luma(const std::uint8_t* bgr, int pixel) {
    const int i = pixel * 3;
    const float g = 0.114f * bgr[i] + 0.587f * bgr[i + 1] + 0.299f * bgr[i + 2];
    return static_cast<std::uint8_t>(g + 0.5f);
}

// ---------------------------------------------------------------- Stage 1
__global__ void gaussianKernel(const std::uint8_t* __restrict__ bgr, std::uint8_t* __restrict__ gray,
                               int width, int height) {
    __shared__ std::uint8_t tile[kBlockY + 2][kBlockX + 2];
    const int x0 = blockIdx.x * kBlockX;
    const int y0 = blockIdx.y * kBlockY;

    // Cooperative load of (kBlockY+2) x (kBlockX+2) tile; clamp-to-edge handles image borders.
    const int tid = threadIdx.y * kBlockX + threadIdx.x;
    for (int i = tid; i < (kBlockX + 2) * (kBlockY + 2); i += kBlockX * kBlockY) {
        const int ty = i / (kBlockX + 2);
        const int tx = i % (kBlockX + 2);
        const int gx = clampi(x0 + tx - 1, 0, width - 1);
        const int gy = clampi(y0 + ty - 1, 0, height - 1);
        tile[ty][tx] = luma(bgr, gy * width + gx);
    }
    __syncthreads();

    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;
    if (x >= width || y >= height) return;

    float acc = 0.0f;
#pragma unroll
    for (int dy = 0; dy < 3; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 3; ++dx) {
            acc += c_gauss[dy * 3 + dx] * static_cast<float>(tile[threadIdx.y + dy][threadIdx.x + dx]);
        }
    }
    gray[y * width + x] = static_cast<std::uint8_t>(acc + 0.5f);
}

// ---------------------------------------------------------------- Stage 2
__global__ void laplacianKernel(const std::uint8_t* __restrict__ gray, std::uint8_t* __restrict__ edge,
                                int width, int height) {
    __shared__ std::uint8_t tile[kBlockY + 2][kBlockX + 2];
    const int x0 = blockIdx.x * kBlockX;
    const int y0 = blockIdx.y * kBlockY;

    const int tid = threadIdx.y * kBlockX + threadIdx.x;
    for (int i = tid; i < (kBlockX + 2) * (kBlockY + 2); i += kBlockX * kBlockY) {
        const int ty = i / (kBlockX + 2);
        const int tx = i % (kBlockX + 2);
        const int gx = clampi(x0 + tx - 1, 0, width - 1);
        const int gy = clampi(y0 + ty - 1, 0, height - 1);
        tile[ty][tx] = gray[gy * width + gx];
    }
    __syncthreads();

    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;
    if (x >= width || y >= height) return;

    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    int response = tile[ty - 1][tx] + tile[ty + 1][tx] + tile[ty][tx - 1] + tile[ty][tx + 1] - 4 * tile[ty][tx];
    response = response < 0 ? -response : response;
    edge[y * width + x] = static_cast<std::uint8_t>(response > 255 ? 255 : response);
}

// ---------------------------------------------------------------- Stage 3
// Labels form a forest: label[i] <= i, roots satisfy label[i] == i, non-edge pixels hold kNoLabel.
__device__ __forceinline__ int findRoot(volatile int* label, int i) {
    int parent = label[i];
    while (parent != i) {
        i = parent;
        parent = label[i];
    }
    return i;
}

__global__ void initLabelsKernel(const std::uint8_t* __restrict__ edge, int* __restrict__ label, int width,
                                 int height, std::uint8_t threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;
    label[i] = edge[i] >= threshold ? i : kNoLabel;
}

/// Each edge pixel links its root to the roots of its 8 neighbours (always the larger
/// root under the smaller one, via atomicMin, so the forest stays acyclic).
__global__ void hookKernel(int* label, int width, int height, int* changed) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;
    if (label[i] == kNoLabel) return;

    int rootI = findRoot(label, i);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const int j = ny * width + nx;
            if (label[j] == kNoLabel) continue;
            const int rootJ = findRoot(label, j);
            if (rootJ == rootI) continue;
            atomicMin(&label[max(rootI, rootJ)], min(rootI, rootJ));
            *changed = 1;
            rootI = findRoot(label, i);
        }
    }
}

/// Pointer jumping: point every pixel directly at its current root.
__global__ void compressKernel(int* label, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;
    if (label[i] != kNoLabel) label[i] = findRoot(label, i);
}

__global__ void accumulateKernel(const int* __restrict__ label, int width, int height,
                                 std::uint32_t* count, unsigned long long* sumX, unsigned long long* sumY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int root = label[y * width + x];
    if (root == kNoLabel) return;
    atomicAdd(&count[root], 1u);
    atomicAdd(&sumX[root], static_cast<unsigned long long>(x));
    atomicAdd(&sumY[root], static_cast<unsigned long long>(y));
}

/// One thread per pixel; only root pixels of large-enough components emit a Detection.
__global__ void collectKernel(const int* __restrict__ label, int width, int height,
                              const std::uint32_t* __restrict__ count, const unsigned long long* __restrict__ sumX,
                              const unsigned long long* __restrict__ sumY, std::uint32_t minArea,
                              Detection* out, int* objectCounter, int maxObjects) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;
    if (label[i] != i) return;
    const std::uint32_t area = count[i];
    if (area < minArea) return;
    const int slot = atomicAdd(objectCounter, 1);
    if (slot < maxObjects) {
        out[slot] = Detection{static_cast<float>(sumX[i]) / static_cast<float>(area),
                              static_cast<float>(sumY[i]) / static_cast<float>(area), area};
    }
}

__device__ __forceinline__ void paint(std::uint8_t* bgr, int i, std::uint8_t b, std::uint8_t g, std::uint8_t r) {
    bgr[i * 3] = b;
    bgr[i * 3 + 1] = g;
    bgr[i * 3 + 2] = r;
}

/// Colours every edge pixel of a reported component; the colour is a hash of the component's root.
__global__ void overlayKernel(std::uint8_t* __restrict__ bgr, const int* __restrict__ label, int width, int height,
                              const std::uint32_t* __restrict__ count, std::uint32_t minArea) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;
    const int root = label[i];
    if (root == kNoLabel || count[root] < minArea) return;

    // Saturated palette in BGR order: red, green, yellow, cyan, magenta, orange.
    const std::uint8_t palette[6][3] = {{0, 0, 255}, {0, 255, 0}, {0, 255, 255},
                                        {255, 255, 0}, {255, 0, 255}, {0, 128, 255}};
    const unsigned pick = (static_cast<unsigned>(root) * 2654435761u >> 16) % 6u;
    paint(bgr, i, palette[pick][0], palette[pick][1], palette[pick][2]);
}

/// Draws a white cross on each centroid. One block per detection.
__global__ void markerKernel(std::uint8_t* bgr, const Detection* detections, int count, int width, int height) {
    const int k = blockIdx.x;
    const int t = threadIdx.x;
    if (k >= count || t > 2 * kCrossRadius) return;
    const int cx = __float2int_rn(detections[k].cx);
    const int cy = __float2int_rn(detections[k].cy);
    const int hx = cx + t - kCrossRadius;
    const int vy = cy + t - kCrossRadius;
    if (hx >= 0 && hx < width && cy >= 0 && cy < height) paint(bgr, cy * width + hx, 255, 255, 255);
    if (vy >= 0 && vy < height && cx >= 0 && cx < width) paint(bgr, vy * width + cx, 255, 255, 255);
}

inline dim3 gridFor(int width, int height) {
    return dim3(static_cast<unsigned>((width + kBlockX - 1) / kBlockX),
                static_cast<unsigned>((height + kBlockY - 1) / kBlockY));
}

}  // namespace detail

// =================================================================== host API
void setGaussianSigma(float sigma) {
    if (!(sigma > 0.0f)) throw std::invalid_argument("sigma must be > 0");
    const std::array<float, 9> k = makeGaussianKernel3x3(sigma);
    IPP_CUDA_CHECK(cudaMemcpyToSymbol(detail::c_gauss, k.data(), sizeof(float) * 9));
}

void launchGaussian(const std::uint8_t* dBgr, std::uint8_t* dGray, int width, int height, cudaStream_t stream) {
    detail::gaussianKernel<<<detail::gridFor(width, height), dim3(detail::kBlockX, detail::kBlockY), 0, stream>>>(
        dBgr, dGray, width, height);
    IPP_CUDA_CHECK(cudaGetLastError());
}

void launchLaplacian(const std::uint8_t* dGray, std::uint8_t* dEdge, int width, int height, cudaStream_t stream) {
    detail::laplacianKernel<<<detail::gridFor(width, height), dim3(detail::kBlockX, detail::kBlockY), 0, stream>>>(
        dGray, dEdge, width, height);
    IPP_CUDA_CHECK(cudaGetLastError());
}

ComponentAnalyzer::ComponentAnalyzer(std::size_t maxPixels, int maxObjects)
    : capacity_(maxPixels), maxObjects_(maxObjects) {
    if (maxPixels == 0 || maxObjects <= 0) throw std::invalid_argument("ComponentAnalyzer: bad capacity");
    label_.allocate(maxPixels);
    count_.allocate(maxPixels);
    sumX_.allocate(maxPixels);
    sumY_.allocate(maxPixels);
    detections_.allocate(static_cast<std::size_t>(maxObjects));
    flags_.allocate(2);
    hostFlags_.allocate(2);
    hostDetections_.allocate(static_cast<std::size_t>(maxObjects));
}

AnalysisResult ComponentAnalyzer::analyze(const std::uint8_t* dEdge, std::uint8_t* dBgr, int width, int height,
                                          std::uint8_t threshold, std::uint32_t minArea, cudaStream_t stream) {
    using namespace detail;
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (width <= 0 || height <= 0 || pixels > capacity_) {
        throw std::invalid_argument("ComponentAnalyzer::analyze: image exceeds analyzer capacity");
    }
    const dim3 grid = gridFor(width, height);
    const dim3 block(kBlockX, kBlockY);
    int* const dChanged = flags_.get();
    int* const dObjectCount = flags_.get() + 1;

    IPP_CUDA_CHECK(cudaMemsetAsync(flags_.get(), 0, 2 * sizeof(int), stream));
    initLabelsKernel<<<grid, block, 0, stream>>>(dEdge, label_.get(), width, height, threshold);
    IPP_CUDA_CHECK(cudaGetLastError());

    // Iterate hook+compress until no pixel changed a root. The host only looks at
    // convergence every kHooksPerCheck iterations to keep round-trips low.
    int rounds = 0;
    for (;;) {
        IPP_CUDA_CHECK(cudaMemsetAsync(dChanged, 0, sizeof(int), stream));
        for (int k = 0; k < kHooksPerCheck; ++k) {
            hookKernel<<<grid, block, 0, stream>>>(label_.get(), width, height, dChanged);
            compressKernel<<<grid, block, 0, stream>>>(label_.get(), width, height);
        }
        IPP_CUDA_CHECK(cudaGetLastError());
        IPP_CUDA_CHECK(cudaMemcpyAsync(hostFlags_.get(), dChanged, sizeof(int), cudaMemcpyDeviceToHost, stream));
        IPP_CUDA_CHECK(cudaStreamSynchronize(stream));
        ++rounds;
        if (hostFlags_.get()[0] == 0) break;
        if (rounds > kMaxLabelingRounds) throw std::runtime_error("connected-component labeling did not converge");
    }

    IPP_CUDA_CHECK(cudaMemsetAsync(count_.get(), 0, pixels * sizeof(std::uint32_t), stream));
    IPP_CUDA_CHECK(cudaMemsetAsync(sumX_.get(), 0, pixels * sizeof(unsigned long long), stream));
    IPP_CUDA_CHECK(cudaMemsetAsync(sumY_.get(), 0, pixels * sizeof(unsigned long long), stream));
    accumulateKernel<<<grid, block, 0, stream>>>(label_.get(), width, height, count_.get(), sumX_.get(), sumY_.get());
    collectKernel<<<grid, block, 0, stream>>>(label_.get(), width, height, count_.get(), sumX_.get(), sumY_.get(),
                                              minArea, detections_.get(), dObjectCount, maxObjects_);
    IPP_CUDA_CHECK(cudaGetLastError());

    // Only the object counter and the (small, fixed-size) detection list travel back to the host.
    IPP_CUDA_CHECK(cudaMemcpyAsync(hostFlags_.get() + 1, dObjectCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
    IPP_CUDA_CHECK(cudaMemcpyAsync(hostDetections_.get(), detections_.get(),
                                   static_cast<std::size_t>(maxObjects_) * sizeof(Detection),
                                   cudaMemcpyDeviceToHost, stream));
    overlayKernel<<<grid, block, 0, stream>>>(dBgr, label_.get(), width, height, count_.get(), minArea);
    IPP_CUDA_CHECK(cudaGetLastError());
    IPP_CUDA_CHECK(cudaStreamSynchronize(stream));

    AnalysisResult result;
    result.labelingRounds = rounds;
    result.totalObjects = static_cast<std::uint32_t>(hostFlags_.get()[1]);
    const int stored = std::min<int>(hostFlags_.get()[1], maxObjects_);
    result.truncated = hostFlags_.get()[1] > maxObjects_;
    result.objects.assign(hostDetections_.get(), hostDetections_.get() + stored);
    std::sort(result.objects.begin(), result.objects.end(), [](const Detection& a, const Detection& b) {
        return a.cy != b.cy ? a.cy < b.cy : a.cx < b.cx;
    });

    if (stored > 0) {
        markerKernel<<<stored, 32, 0, stream>>>(dBgr, detections_.get(), stored, width, height);
        IPP_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

}  // namespace ipp
